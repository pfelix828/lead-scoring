import Link from "next/link";
import { framework, meta, model, targets } from "@/lib/data";
import { num, pct } from "@/lib/format";
import { Card, CardTitle, PageHeader, Stat, SyntheticDataNote, Term } from "@/components/ui";
import { QuadrantScatter } from "@/components/charts";

const QUADRANT_CELLS = [
  { key: "high_prop_high_complete", title: "Likely to buy, group complete", action: "Work these now — sales-ready." },
  { key: "high_prop_low_complete", title: "Likely to buy, group incomplete", action: "Enrich first: highest-ROI target list." },
  { key: "low_prop_high_complete", title: "Unlikely, group complete", action: "Nurture; the people are there, the fit isn't yet." },
  { key: "low_prop_low_complete", title: "Unlikely, group incomplete", action: "Deprioritize." },
] as const;

export default function FrameworkPage() {
  const q = framework.quadrants;
  return (
    <div className="space-y-6">
      <PageHeader
        title="Which accounts do we work, and what do we fix at each one?"
        subtitle={
          <>
            Two scores answer that together. A <Term def="Logistic regression on pre-deal features only: firmographics, technographics, and contact composition. Trained on ~29K accounts with closed deals, evaluated on a held-out 7K.">propensity score</Term>{" "}
            says how likely an account is to buy. A{" "}
            <Term def="0-100 score across four dimensions (role coverage, seniority mix, function diversity, technical+business pairing) measuring whether the right people are on the deal.">buying-group completeness score</Term>{" "}
            says whether the right people are at the table. Crossing them turns 100K accounts into four lists with
            different next actions.
          </>
        }
      />

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardTitle sub={`One dot per account (1,500-account sample of the ${num(framework.scatter.length > 0 ? meta.with_contacts_n : 0)} scored accounts with contacts; purple = won). Dashed lines split the four quadrants.`}>
            The 2×2 that runs the play
          </CardTitle>
          <QuadrantScatter scatter={framework.scatter} median={framework.median_propensity} />
        </Card>

        <div className="space-y-4">
          {QUADRANT_CELLS.map((cell) => (
            <Card key={cell.key} className={cell.key === "high_prop_low_complete" ? "border-accent/50" : undefined}>
              <div className="flex items-baseline justify-between gap-2">
                <h3 className="text-sm font-semibold tracking-tight">{cell.title}</h3>
                <span className="text-lg font-bold tabular-nums tracking-tight">{q[cell.key].toLocaleString()}</span>
              </div>
              <p className="mt-1 text-xs text-muted">{cell.action}</p>
              {cell.key === "high_prop_low_complete" ? (
                <Link href="/targets" className="mt-2 inline-block text-xs font-medium text-accent underline-offset-2 hover:underline">
                  See the target list →
                </Link>
              ) : null}
            </Card>
          ))}
        </div>
      </div>

      <Card>
        <CardTitle>The one read</CardTitle>
        <p className="max-w-3xl text-sm leading-relaxed">
          {targets.summary.n_targets.toLocaleString()} accounts score above median propensity but have incomplete
          buying groups — the model says they&apos;re worth working, and the contact data says why they stall. Filling
          named role gaps at those accounts is cheaper than finding new demand, which makes this quadrant the
          highest-ROI list the framework produces. Where the numbers behind both axes come from — and how seriously
          to take them — is on the{" "}
          <Link href="/model" className="text-accent underline-offset-2 hover:underline">
            Model
          </Link>{" "}
          and{" "}
          <Link href="/buying-groups" className="text-accent underline-offset-2 hover:underline">
            Buying Groups
          </Link>{" "}
          pages.
        </p>
      </Card>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <Stat label="Accounts in base" value={num(meta.accounts_total)} hint="Synthetic CRM: accounts, contacts, opportunities, contact-deal roles." />
        </Card>
        <Card>
          <Stat
            label={<Term def={`Bootstrap 95% CI ${model.metrics.auc_ci_lower.toFixed(2)}-${model.metrics.auc_ci_upper.toFixed(2)} on the held-out test set.`}>Test AUC</Term>}
            value={model.metrics.auc.toFixed(3)}
            hint="Deliberately modest — see the Model page for why that's realistic and what it's still good for."
          />
        </Card>
        <Card>
          <Stat
            label="Precision in top decile"
            value={pct(model.metrics.precision_at_10pct)}
            hint={`vs ${pct(meta.win_rate_overall)} base win rate; CI ${pct(model.metrics.precision_at_10pct_ci_lower)}-${pct(model.metrics.precision_at_10pct_ci_upper)}.`}
          />
        </Card>
      </div>

      <SyntheticDataNote />
    </div>
  );
}
