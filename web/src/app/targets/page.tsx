import { targets } from "@/lib/data";
import { pct } from "@/lib/format";
import { Card, CardTitle, PageHeader, Stat, SyntheticDataNote } from "@/components/ui";
import { TargetTable } from "@/components/target-table";

export default function TargetsPage() {
  const s = targets.summary;
  return (
    <div className="space-y-6">
      <PageHeader
        title="The enrichment target list"
        subtitle={
          <>
            Accounts the model already likes (propensity above the {pct(s.median_propensity_threshold, 1)} median)
            whose buying groups are incomplete (completeness below 50, with at least some contacts). The next action
            is specific and cheap relative to demand generation: add the named missing roles via enrichment vendors
            or targeted outreach, then re-score.
          </>
        }
      />

      <div className="grid gap-4 sm:grid-cols-4">
        <Card><Stat label="Target accounts" value={s.n_targets.toLocaleString()} hint="High propensity, incomplete group, has contacts." /></Card>
        <Card><Stat label="Broader gap pool" value={s.gap_pool_n.toLocaleString()} hint="All accounts with completeness ≤ 75 and contacts, across the full 100K base — the enrichment universe beyond the priority list." /></Card>
        <Card><Stat label="Avg propensity" value={pct(s.avg_propensity, 1)} hint="Mean model score across target accounts." /></Card>
        <Card><Stat label="Avg completeness" value={`${s.avg_completeness}`} hint="Mean buying-group score (0-100) across target accounts." /></Card>
      </div>

      <Card>
        <CardTitle sub="Top 30 by propensity. Filter by segment; amber chips are the roles to add.">
          Work the list
        </CardTitle>
        <TargetTable />
        <p className="mt-3 text-xs text-muted">
          In production this table is a CRM view or a CSV handed to an enrichment vendor with role filters attached —
          the demo shows the top 30 of {s.n_targets.toLocaleString()}.
        </p>
      </Card>

      <SyntheticDataNote />
    </div>
  );
}
