/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2024. All rights reserved.
 * Description: Jank-aware frequency scaler: uses MISC-like baseline from kernel
 *   target_freq, then amplifies 0.5x/0.75x/1.0x/1.25x based on last 5 frames jank.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <lib/utils.h>
#include <hongmeng/errno.h>
#include <libhmlog/hmlog.h>
#include <libhmsync/raw_thread.h>
#include <libhmsync/raw_mutex.h>
#include <hmkernel/power/sched_indicator.h>

#include "dfc_tuner.h"
#include "freqmgr/driver/dfc_driver.h"
#include "sched_ind_listener.h"
#include "common/pm_notifier.h"
#include "misc/misc_features_snapshot.h"

#define QLM_JANK_WINDOW 5

static const unsigned int qlm_scale[QLM_JANK_WINDOW + 1] = {
	125, /* 0 jank-free → 1.25x: jank just happened, boost */
	100, /* 1 jank-free → 1.00x */
	 75, /* 2 jank-free → 0.75x */
	 75, /* 3 jank-free → 0.75x */
	 50, /* 4 jank-free → 0.50x */
	 50, /* 5 jank-free → 0.50x */
};

struct tuner_priv {
	unsigned int tuner_id;
	struct dfc_prop *prop;
	unsigned int cluster_id;

	/* Jank history ring buffer */
	unsigned long long prev_jank;
	int jank_free_count;
	int jank_window[QLM_JANK_WINDOW];
	int jank_idx;

	/* Listeners for sched_ind data */
	struct pm_listener load_change_listener;
	unsigned int target_freq;

	struct raw_mutex lock;
};

static int handle_load_change(const struct pm_listener *listener, void *data)
{
	struct tuner_priv *priv;
	struct __sched_ind_load_chg_data *chg = (struct __sched_ind_load_chg_data *)data;
	unsigned long long cur_jank;
	int has_jank;
	unsigned int scale;
	unsigned int new_freq;

	if (data == NULL)
		return E_HM_INVAL;

	priv = container_of(listener, struct tuner_priv, load_change_listener);
	cur_jank = chg->features.janky_frames;

	raw_mutex_lock(&priv->lock);

	/* Track jank over rolling window */
	has_jank = (cur_jank > priv->prev_jank) ? 1 : 0;
	priv->jank_window[priv->jank_idx] = has_jank;
	priv->jank_idx = (priv->jank_idx + 1) % QLM_JANK_WINDOW;

	/* Count jank-free frames in window */
	priv->jank_free_count = 0;
	for (int i = 0; i < QLM_JANK_WINDOW; i++)
		if (priv->jank_window[i] == 0)
			priv->jank_free_count++;

	priv->prev_jank = cur_jank;

	/* Get baseline from kernel target freq */
	priv->target_freq = chg->clusters_data[priv->cluster_id].target_freq;
	if (priv->target_freq == 0)
		priv->target_freq = priv->prop->min_freq;

	/* Apply scale */
	scale = qlm_scale[priv->jank_free_count];
	new_freq = (unsigned int)((unsigned long long)priv->target_freq * scale / 100);

	/* Clamp to hardware limits */
	if (new_freq < priv->prop->min_freq)
		new_freq = priv->prop->min_freq;
	if (new_freq > priv->prop->max_freq)
		new_freq = priv->prop->max_freq;

	dfc_driver_set_freq(priv->prop, new_freq, FREQ_TABLE_CEIL_METHOD);

	/* --- Trace collection -------------------------------------------------
	 * Emit one parseable line per event so real device traces can be harvested
	 * for offline hyperparameter tuning (feed them to tunner/dfc_qlearn_sim.c).
	 * The four features mirror struct __sched_ind_qlearn_features and match the
	 * simulator: load / refresh / power / frame_budget, plus the jank counter
	 * and the frequency we set. Harvest with:  dmesg | grep ' dfs '
	 * NOTE: verify the exact field names against the kernel header for your
	 * build (cluster[] vs clusters_data[], member spellings).
	 */
	{
		struct __sched_ind_qlearn_features *qf = &chg->features;
		char term[160];
		snprintf(term, sizeof(term),
			"load=%u rr=%u power=%u fbud=%d jank=%llu freq=%u",
			qf->cluster[priv->cluster_id].avg_load,
			qf->curr_refresh_rate,
			qf->cluster[priv->cluster_id].curr_power,
			qf->frame_budget,
			cur_jank,
			new_freq);
		hm_error("dfs %s\n", term);
	}

	hm_error("QLM: target=%u scale=%u%% new=%u jfree=%d/%d jank=%llu\n",
		 priv->target_freq, scale, new_freq,
		 priv->jank_free_count, QLM_JANK_WINDOW, cur_jank);

	raw_mutex_unlock(&priv->lock);
	return E_HM_OK;
}

static unsigned int *qlm_init(struct dfc_prop *prop)
{
	struct tuner_priv *priv;

	priv = (struct tuner_priv *)malloc(sizeof(*priv));
	if (priv == NULL) {
		hm_error("alloc qlm priv failed\n");
		return NULL;
	}

	memset(priv, 0, sizeof(*priv));
	priv->prop = prop;
	priv->cluster_id = dfc_cpu_prop_of(prop)->cluster_id;
	priv->prev_jank = 0;
	priv->jank_free_count = QLM_JANK_WINDOW;
	priv->jank_idx = 0;
	priv->target_freq = prop->min_freq;
	priv->lock = RAW_MUTEX_INITIALIZER;

	/* Register sched_ind load change listener */
	priv->load_change_listener.event = PM_NOTIFIER_LOAD_CHANGE;
	priv->load_change_listener.event_handler = handle_load_change;

	hm_error("QLMINIT: cluster=%u min=%u max=%u\n",
		 priv->cluster_id, prop->min_freq, prop->max_freq);

	return &priv->tuner_id;
}

static void qlm_destroy(struct dfc_prop *prop)
{
	struct tuner_priv *priv;

	if (prop->tuner_id_ctx == NULL)
		return;

	priv = container_of(prop->tuner_id_ctx, struct tuner_priv, tuner_id);

	hm_error("QLMDESTROY: cluster=%u\n", priv->cluster_id);

	free(priv);
	prop->tuner_id_ctx = NULL;
}

static int qlm_start(struct dfc_prop *prop, unsigned int *tuner_id_ctx)
{
	struct tuner_priv *priv;
	int err;

	if (prop == NULL)
		return E_HM_INVAL;

	priv = container_of(tuner_id_ctx, struct tuner_priv, tuner_id);
	priv->prop = prop;

	/* Add listener to receive sched_ind data */
	err = pm_notifier_add_listener(&priv->load_change_listener);
	if (err != E_HM_OK) {
		hm_warn("qlm add listener failed: %s\n", hmstrerror(err));
		return err;
	}

	/* Start with min freq, let first load change event adjust it */
	err = dfc_driver_set_freq(prop, prop->min_freq, FREQ_TABLE_CEIL_METHOD);
	if (err != E_HM_OK) {
		hm_warn("qlm start freq failed: %s\n", hmstrerror(err));
		return err;
	}

	hm_error("QLMSTART: cluster=%u min=%u listener=ok\n",
		 priv->cluster_id, prop->min_freq);

	return E_HM_OK;
}

static void qlm_stop(struct dfc_prop *prop)
{
	struct tuner_priv *priv;

	if (prop->tuner_id_ctx == NULL)
		return;

	priv = container_of(prop->tuner_id_ctx, struct tuner_priv, tuner_id);

	hm_error("QLMSTOP: cluster=%u\n", priv->cluster_id);
}

static int qlm_reset(struct dfc_prop *prop)
{
	int err;

	if (prop == NULL)
		return E_HM_INVAL;

	err = dfc_driver_set_freq(prop, prop->cur_freq, FREQ_TABLE_CEIL_METHOD);
	if (err != E_HM_OK) {
		hm_warn("qlm reset freq failed: %s\n", hmstrerror(err));
		return err;
	}

	return E_HM_OK;
}

static struct dfc_tuner dfc_tuner_qlearn_misc = {
	.type = DFC_TUNER_TYPE_QLEARN_MISC,
	.name = DFC_TUNER_QLEARN_MISC_NAME,
	.alias_name = "qlm",
	.init = qlm_init,
	.destroy = qlm_destroy,
	.start = qlm_start,
	.stop = qlm_stop,
	.reset = qlm_reset,
};

dfc_tuner_constructor(qlearn_misc, &dfc_tuner_qlearn_misc, DFC_COMP_TYPE_CPU)
