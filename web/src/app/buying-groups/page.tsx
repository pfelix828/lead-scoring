import { framework, targets } from "@/lib/data";
import { Card, CardTitle, PageHeader, SeverityBadge, SyntheticDataNote, Term } from "@/components/ui";
import { CountBars, SubScoreBars, TierBars } from "@/components/charts";

export default function BuyingGroupsPage() {
  const tiers = framework.tiers;
  const ratio = tiers[3].win_rate / tiers[0].win_rate;
  return (
    <div className="space-y-6">
      <PageHeader
        title="Deals are won by groups, not leads"
        subtitle={
          <>
            B2B purchases involve a buying group: an economic buyer, technical evaluators, business champions. The{" "}
            <Term def="0-100 = sum of four 0-25 dimensions: role coverage, seniority mix, function diversity, technical+business pairing.">
              completeness score
            </Term>{" "}
            measures whether that group is actually present in the CRM — and which specific people are missing.
          </>
        }
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardTitle sub="Win rate by completeness tier, all 35,973 accounts with closed deals. Hover for account counts.">
            Complete groups close {ratio.toFixed(1)}× more
          </CardTitle>
          <TierBars tiers={tiers} />
        </Card>
        <Card className="border-amber-300/70">
          <div className="mb-2 flex items-center gap-2">
            <SeverityBadge severity="watch" />
            <h3 className="text-sm font-semibold tracking-tight">Read the {ratio.toFixed(1)}× as association, not cause</h3>
          </div>
          <p className="text-sm leading-relaxed text-muted">
            Complete buying groups co-occur with larger, more engaged, later-stage accounts that were likelier to win
            anyway — so the raw gap overstates what filling contact gaps would cause. Isolating the causal effect
            would take a controlled test (randomize which gap accounts get enriched) or, short of that, adjustment
            for account size, segment, and deal stage.
          </p>
          <p className="mt-3 text-sm leading-relaxed text-muted">
            The framework doesn&apos;t need the causal claim to be useful: completeness still tells you{" "}
            <em>which specific roles are absent</em> at accounts the model already likes — a concrete enrichment
            action with a bounded cost, sequenced ahead of unmeasurable brand spend.
          </p>
        </Card>
      </div>

      <Card>
        <CardTitle sub="Average score on each 0-25 dimension, by segment, across accounts with contacts.">
          Where groups fall short, by segment
        </CardTitle>
        <SubScoreBars rows={framework.sub_scores} />
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted">
          Function diversity holds up across segments; seniority mix and the technical+business pairing are where
          SMB and Mid-Market accounts thin out — consistent with smaller companies simply having fewer titles to
          add, which is why the target list weights <em>which</em> role is missing, not just how many.
        </p>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardTitle sub={`Most common missing roles across the ${targets.summary.n_targets.toLocaleString()} high-propensity gap accounts.`}>
            The specific people to add
          </CardTitle>
          <CountBars rows={targets.summary.missing_roles.map((r) => ({ label: r.role, n: r.n }))} />
        </Card>
        <Card>
          <CardTitle sub="Structural absences in the same gap accounts — gaps that block deals regardless of count.">
            Structural gaps
          </CardTitle>
          <CountBars rows={targets.summary.structural.map((r) => ({ label: r.gap, n: r.n }))} color="#b45309" />
          <p className="mt-2 text-sm leading-relaxed text-muted">
            No VP+ means no economic buyer on record; no technical contact means evaluation stalls. These two
            absences are the highest-value single adds.
          </p>
        </Card>
      </div>

      <SyntheticDataNote />
    </div>
  );
}
