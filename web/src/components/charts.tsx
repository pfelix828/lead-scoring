"use client";

/** Bespoke Recharts views. Each chart answers one question, stated in its
 *  card title, with the takeaway in surrounding copy rather than left for
 *  the reader to infer. */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  ErrorBar,
  LabelList,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { featureLabel, type BaselineRow, type CalibrationBin, type ImportanceRow, type LiftRow } from "@/lib/data";
import { pct } from "@/lib/format";

const ACCENT = "#4f46e5";
const GRAY = "#9ca3af";
const LIGHT = "#c7d2fe";
const RED = "#b91c1c";
const GREEN = "#047857";

/** The 2x2: completeness (x) vs propensity (y), one dot per account. */
export function QuadrantScatter({
  scatter,
  median,
}: {
  scatter: { x: number; y: number; won: number }[];
  median: number;
}) {
  const won = scatter.filter((p) => p.won === 1);
  const lost = scatter.filter((p) => p.won === 0);
  return (
    <ResponsiveContainer width="100%" height={380}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
        <CartesianGrid stroke="#eee" />
        <XAxis
          type="number"
          dataKey="x"
          domain={[0, 100]}
          tick={{ fontSize: 11, fill: "#6b7280" }}
          tickLine={false}
          axisLine={{ stroke: "#e4e4e7" }}
          label={{ value: "Buying-group completeness (0-100)", position: "insideBottom", offset: -4, fontSize: 11, fill: "#6b7280" }}
        />
        <YAxis
          type="number"
          dataKey="y"
          domain={[0, 0.8]}
          tickFormatter={(v) => pct(v)}
          tick={{ fontSize: 11, fill: "#6b7280" }}
          tickLine={false}
          axisLine={false}
          label={{ value: "Propensity score", angle: -90, position: "insideLeft", fontSize: 11, fill: "#6b7280" }}
        />
        <ReferenceLine x={50} stroke="#9ca3af" strokeDasharray="5 4" />
        <ReferenceLine y={median} stroke="#9ca3af" strokeDasharray="5 4" label={{ value: "median propensity", position: "insideTopRight", fontSize: 10, fill: "#9ca3af" }} />
        <Tooltip
          formatter={(value, name) => [name === "y" ? pct(Number(value), 1) : value, name === "y" ? "propensity" : "completeness"]}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e4e4e7" }}
        />
        <Scatter data={lost} fill={GRAY} fillOpacity={0.35} shape="circle" name="lost" />
        <Scatter data={won} fill={ACCENT} fillOpacity={0.65} shape="circle" name="won" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/** Naive baselines vs the full model, with bootstrap 95% CIs. */
export function BaselinesLadder({ baselines }: { baselines: BaselineRow[] }) {
  const rows = baselines.map((b) => ({
    name: b.model,
    auc: b.auc,
    err: [b.auc - b.auc_ci_lower, b.auc_ci_upper - b.auc],
    full: b.model.startsWith("Full"),
  }));
  return (
    <ResponsiveContainer width="100%" height={rows.length * 52 + 30}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 48, bottom: 0, left: 56 }}>
        <CartesianGrid stroke="#eee" horizontal={false} />
        <XAxis type="number" domain={[0.45, 0.62]} tickFormatter={(v) => v.toFixed(2)} tick={{ fontSize: 11, fill: "#6b7280" }} tickLine={false} axisLine={false} />
        <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 11.5, fill: "#374151" }} tickLine={false} axisLine={false} />
        <ReferenceLine x={0.5} stroke="#9ca3af" strokeDasharray="5 4" label={{ value: "coin flip", position: "top", fontSize: 10, fill: "#9ca3af" }} />
        <Tooltip
          formatter={(value) => [Number(value).toFixed(3), "test AUC"]}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e4e4e7" }}
        />
        <Bar dataKey="auc" barSize={16} radius={[0, 3, 3, 0]}>
          {rows.map((r) => (
            <Cell key={r.name} fill={r.full ? ACCENT : LIGHT} />
          ))}
          <LabelList dataKey="auc" position="right" formatter={(v) => Number(v).toFixed(3)} style={{ fontSize: 11, fill: "#374151" }} />
          <ErrorBar dataKey="err" direction="x" width={4} strokeWidth={1.5} stroke="#374151" />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Predicted vs observed win rate by score decile. */
export function CalibrationChart({ bins }: { bins: CalibrationBin[] }) {
  const max = Math.max(...bins.map((b) => Math.max(b.predicted, b.actual))) * 1.15;
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
        <CartesianGrid stroke="#eee" />
        <XAxis
          type="number"
          dataKey="predicted"
          domain={[0, max]}
          tickFormatter={(v) => pct(v)}
          tick={{ fontSize: 11, fill: "#6b7280" }}
          tickLine={false}
          axisLine={{ stroke: "#e4e4e7" }}
          label={{ value: "Predicted win rate", position: "insideBottom", offset: -4, fontSize: 11, fill: "#6b7280" }}
        />
        <YAxis
          type="number"
          dataKey="actual"
          domain={[0, max]}
          tickFormatter={(v) => pct(v)}
          tick={{ fontSize: 11, fill: "#6b7280" }}
          tickLine={false}
          axisLine={false}
          label={{ value: "Observed win rate", angle: -90, position: "insideLeft", fontSize: 11, fill: "#6b7280" }}
        />
        <ReferenceLine
          segment={[
            { x: 0, y: 0 },
            { x: max, y: max },
          ]}
          stroke="#9ca3af"
          strokeDasharray="5 4"
          label={{ value: "perfectly calibrated", position: "insideTopLeft", fontSize: 10, fill: "#9ca3af" }}
        />
        <Tooltip
          formatter={(value, name) => [pct(Number(value), 1), name === "predicted" ? "predicted" : "observed"]}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e4e4e7" }}
        />
        <Scatter data={bins} fill={ACCENT} line={{ stroke: ACCENT, strokeWidth: 1.5 }} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/** Score distributions for winners vs losers — the honest "weak separator" view. */
export function ScoreHistogram({ hist }: { hist: { bin: number; won: number; lost: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={hist} margin={{ top: 8, right: 8, bottom: 0, left: -16 }} barCategoryGap={1}>
        <CartesianGrid stroke="#eee" vertical={false} />
        <XAxis dataKey="bin" tickFormatter={(v) => pct(Number(v))} tick={{ fontSize: 11, fill: "#6b7280" }} tickLine={false} axisLine={{ stroke: "#e4e4e7" }} minTickGap={24} />
        <YAxis tick={{ fontSize: 11, fill: "#6b7280" }} tickLine={false} axisLine={false} />
        <Tooltip
          labelFormatter={(l) => `score ≈ ${pct(Number(l))}`}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e4e4e7" }}
        />
        <Bar dataKey="lost" name="Lost" stackId="a" fill={GRAY} fillOpacity={0.55} />
        <Bar dataKey="won" name="Won" stackId="a" fill={ACCENT} fillOpacity={0.85} radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Lift per decile (bars) + cumulative share of all wins captured (line). */
export function LiftCaptureChart({ lift }: { lift: LiftRow[] }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={lift} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid stroke="#eee" vertical={false} />
        <XAxis dataKey="decile" tickFormatter={(v) => `D${v}`} tick={{ fontSize: 11, fill: "#6b7280" }} tickLine={false} axisLine={{ stroke: "#e4e4e7" }} />
        <YAxis yAxisId="lift" tick={{ fontSize: 11, fill: "#6b7280" }} tickLine={false} axisLine={false} domain={[0, 1.5]} tickFormatter={(v) => `${v}x`} />
        <YAxis yAxisId="cap" orientation="right" tickFormatter={(v) => pct(v)} tick={{ fontSize: 11, fill: "#6b7280" }} tickLine={false} axisLine={false} domain={[0, 1]} />
        <ReferenceLine yAxisId="lift" y={1} stroke="#9ca3af" strokeDasharray="5 4" label={{ value: "random", position: "insideTopRight", fontSize: 10, fill: "#9ca3af" }} />
        <Tooltip
          formatter={(value, name) =>
            name === "cumulative wins captured" ? [pct(Number(value)), name] : [`${Number(value).toFixed(2)}x`, name]
          }
          labelFormatter={(l) => `Decile ${l} (D1 = highest scores)`}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e4e4e7" }}
        />
        <Bar yAxisId="lift" dataKey="lift" name="lift vs random" fill={LIGHT} radius={[3, 3, 0, 0]} barSize={26}>
          <LabelList dataKey="lift" position="top" formatter={(v) => `${Number(v).toFixed(1)}x`} style={{ fontSize: 10, fill: "#374151" }} />
        </Bar>
        <Line yAxisId="cap" dataKey="cumulative_capture" name="cumulative wins captured" stroke={ACCENT} strokeWidth={2} dot={{ r: 3 }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** Top model coefficients, signed. */
export function ImportanceBars({ importance }: { importance: ImportanceRow[] }) {
  const rows = importance.map((r) => ({ name: featureLabel(r.feature), coef: r.coefficient }));
  return (
    <ResponsiveContainer width="100%" height={rows.length * 26 + 30}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, bottom: 0, left: 64 }}>
        <CartesianGrid stroke="#eee" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 11, fill: "#6b7280" }} tickLine={false} axisLine={false} />
        <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 11, fill: "#374151" }} tickLine={false} axisLine={false} />
        <ReferenceLine x={0} stroke="#d1d5db" />
        <Tooltip
          formatter={(value) => [Number(value).toFixed(3), "coefficient"]}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e4e4e7" }}
        />
        <Bar dataKey="coef" barSize={12} radius={2}>
          {rows.map((r) => (
            <Cell key={r.name} fill={r.coef >= 0 ? GREEN : RED} fillOpacity={0.75} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Win rate by completeness tier. */
export function TierBars({ tiers }: { tiers: { tier: string; win_rate: number; n: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={tiers} margin={{ top: 20, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid stroke="#eee" vertical={false} />
        <XAxis dataKey="tier" tick={{ fontSize: 11, fill: "#374151" }} tickLine={false} axisLine={{ stroke: "#e4e4e7" }} interval={0} />
        <YAxis tickFormatter={(v) => pct(v)} domain={[0, 0.55]} tick={{ fontSize: 11, fill: "#6b7280" }} tickLine={false} axisLine={false} />
        <Tooltip
          formatter={(value) => [pct(Number(value), 1), "win rate"]}
          labelFormatter={(l, payload) => `${l} · ${payload?.[0]?.payload?.n?.toLocaleString()} accounts`}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e4e4e7" }}
        />
        <Bar dataKey="win_rate" barSize={56} radius={[3, 3, 0, 0]}>
          {tiers.map((t, i) => (
            <Cell key={t.tier} fill={ACCENT} fillOpacity={0.35 + i * 0.2} />
          ))}
          <LabelList dataKey="win_rate" position="top" formatter={(v) => pct(Number(v))} style={{ fontSize: 11, fontWeight: 600, fill: "#374151" }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Average buying-group sub-scores by segment (each dimension is 0-25). */
export function SubScoreBars({ rows }: { rows: Record<string, string | number>[] }) {
  const dims = ["Role coverage", "Seniority mix", "Function diversity", "Technical + business"];
  const colors = [ACCENT, "#818cf8", "#a5b4fc", "#6366f1"];
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: -24 }} barGap={2}>
        <CartesianGrid stroke="#eee" vertical={false} />
        <XAxis dataKey="segment" tick={{ fontSize: 11.5, fill: "#374151" }} tickLine={false} axisLine={{ stroke: "#e4e4e7" }} />
        <YAxis domain={[0, 25]} tick={{ fontSize: 11, fill: "#6b7280" }} tickLine={false} axisLine={false} />
        <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e4e4e7" }} />
        {dims.map((d, i) => (
          <Bar key={d} dataKey={d} fill={colors[i]} fillOpacity={0.8} radius={[2, 2, 0, 0]} barSize={18} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Horizontal counts (missing roles, structural gaps). */
export function CountBars({ rows, color = ACCENT }: { rows: { label: string; n: number }[]; color?: string }) {
  return (
    <ResponsiveContainer width="100%" height={rows.length * 32 + 24}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 40, bottom: 0, left: 36 }}>
        <CartesianGrid stroke="#eee" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 11, fill: "#6b7280" }} tickLine={false} axisLine={false} />
        <YAxis type="category" dataKey="label" width={140} tick={{ fontSize: 11.5, fill: "#374151" }} tickLine={false} axisLine={false} />
        <Tooltip
          formatter={(value) => [Number(value).toLocaleString(), "accounts"]}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e4e4e7" }}
        />
        <Bar dataKey="n" fill={color} fillOpacity={0.8} barSize={14} radius={[0, 3, 3, 0]}>
          <LabelList dataKey="n" position="right" formatter={(v) => Number(v).toLocaleString()} style={{ fontSize: 10.5, fill: "#374151" }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
