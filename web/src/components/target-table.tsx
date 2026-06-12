"use client";

/** Filterable enrichment target list. All 30 rows ship in the page; filtering
 *  is display-only. */

import { useMemo, useState } from "react";
import clsx from "clsx";
import { targets } from "@/lib/data";
import { pct } from "@/lib/format";
import { Pill } from "@/components/ui";

export function TargetTable() {
  const [segment, setSegment] = useState<string>("All");
  const segments = useMemo(
    () => ["All", ...Array.from(new Set(targets.rows.map((r) => r.segment))).sort()],
    [],
  );
  const rows = targets.rows.filter((r) => segment === "All" || r.segment === segment);

  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-1.5">
        {segments.map((s) => (
          <button
            key={s}
            onClick={() => setSegment(s)}
            className={clsx(
              "rounded-md px-2.5 py-1.5 text-xs font-medium",
              s === segment ? "bg-accent-soft text-accent" : "bg-zinc-100 text-foreground/70 hover:text-foreground",
            )}
          >
            {s}
          </button>
        ))}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-border-subtle text-xs text-muted">
              <th className="py-2 pr-3 font-medium">Account</th>
              <th className="py-2 pr-3 font-medium">Propensity</th>
              <th className="py-2 pr-3 font-medium">Completeness</th>
              <th className="py-2 pr-3 font-medium">Missing roles</th>
              <th className="py-2 font-medium">VP+ present</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.company} className="border-b border-border-subtle/60 align-top">
                <td className="py-2 pr-3">
                  <div className="font-medium">{r.company}</div>
                  <div className="text-xs text-muted">{r.segment} · {r.industry}</div>
                </td>
                <td className="py-2 pr-3 tabular-nums font-medium">{pct(r.propensity, 1)}</td>
                <td className="py-2 pr-3 tabular-nums">{r.completeness}</td>
                <td className="py-2 pr-3">
                  <div className="flex max-w-md flex-wrap gap-1">
                    {r.missing.slice(0, 4).map((m) => (
                      <Pill key={m} className="border-amber-300/60 bg-amber-50 text-amber-900">{m}</Pill>
                    ))}
                    {r.missing.length > 4 ? <span className="text-xs text-muted">+{r.missing.length - 4} more</span> : null}
                  </div>
                </td>
                <td className="py-2 text-xs">{r.has_vp ? "Yes" : <span className="font-medium text-red-700">No</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
