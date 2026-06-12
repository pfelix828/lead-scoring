/** Typed loaders for the JSON exported by scripts/export_web.py.
 *  Every number was recomputed from the committed seed-42 parquet data
 *  through the same code paths the analysis used (src/model.py etc.). */

import metaJson from "@/data/meta.json";
import modelJson from "@/data/model.json";
import frameworkJson from "@/data/framework.json";
import targetsJson from "@/data/targets.json";

export interface Meta {
  accounts_total: number;
  scored_n: number;
  test_n: number;
  with_contacts_n: number;
  win_rate_overall: number;
}

export interface BaselineRow {
  model: string;
  n_features: number;
  auc: number;
  auc_ci_lower: number;
  auc_ci_upper: number;
}

export interface CalibrationBin {
  predicted: number;
  actual: number;
  count: number;
}

export interface LiftRow {
  decile: number;
  count: number;
  wins: number;
  win_rate: number;
  avg_score: number;
  lift: number;
  cumulative_wins: number;
  cumulative_capture: number;
}

export interface ImportanceRow {
  feature: string;
  coefficient: number;
}

export interface ModelData {
  metrics: {
    auc: number;
    auc_ci_lower: number;
    auc_ci_upper: number;
    log_loss: number;
    precision_at_10pct: number;
    precision_at_20pct: number;
    precision_at_30pct: number;
    precision_at_10pct_ci_lower: number;
    precision_at_10pct_ci_upper: number;
    cv_auc: number;
  };
  baselines: BaselineRow[];
  calibration: {
    bins: CalibrationBin[];
    max_gap: number;
    brier: number;
    brier_no_skill: number;
    test_n: number;
  };
  lift: LiftRow[];
  importance: ImportanceRow[];
  score_hist: { bin: number; won: number; lost: number }[];
}

export interface FrameworkData {
  median_propensity: number;
  quadrants: {
    high_prop_low_complete: number;
    high_prop_high_complete: number;
    low_prop_low_complete: number;
    low_prop_high_complete: number;
  };
  scatter: { x: number; y: number; won: number }[];
  tiers: { tier: string; win_rate: number; n: number }[];
  sub_scores: Record<string, string | number>[];
}

export interface TargetRow {
  company: string;
  segment: string;
  industry: string;
  propensity: number;
  completeness: number;
  missing: string[];
  has_vp: boolean;
}

export interface TargetsData {
  summary: {
    gap_pool_n: number;
    n_targets: number;
    avg_propensity: number;
    avg_completeness: number;
    median_propensity_threshold: number;
    missing_roles: { role: string; n: number }[];
    structural: { gap: string; n: number }[];
  };
  rows: TargetRow[];
}

export const meta = metaJson as Meta;
export const model = modelJson as ModelData;
export const framework = frameworkJson as FrameworkData;
export const targets = targetsJson as TargetsData;

/** Plain-language names for model features. */
const FEATURE_LABELS: Record<string, string> = {
  senior_density: "Senior contact density",
  has_existing_product: "Existing customer",
  max_seniority: "Highest seniority present",
  high_signal_tool_count: "High-signal tools in stack",
  contact_count: "Contacts on account",
  vp_plus_count: "VP+ contacts",
  director_plus_count: "Director+ contacts",
  function_diversity: "Distinct functions",
  has_technical: "Technical contact present",
  has_business: "Business contact present",
  has_technical_and_business: "Technical + business pair",
  employee_count: "Employee count",
  annual_revenue: "Annual revenue",
  tech_overlap_count: "Complementary tech stack",
};

export function featureLabel(feature: string): string {
  if (FEATURE_LABELS[feature]) return FEATURE_LABELS[feature];
  if (feature.startsWith("seg_")) return `Segment: ${feature.slice(4)}`;
  if (feature.startsWith("ind_")) return `Industry: ${feature.slice(4)}`;
  if (feature.startsWith("region_")) return `Region: ${feature.slice(7)}`;
  if (feature.startsWith("tech_")) return `Tech: ${feature.slice(5).replaceAll("_", " ")}`;
  return feature.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
}
