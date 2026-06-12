import { meta, model } from "@/lib/data";
import { Card, CardTitle, PageHeader, Term } from "@/components/ui";

export default function MethodologyPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Methodology"
        subtitle="What the data is, how leakage was kept out, how uncertainty was measured, and exactly how seriously to take the numbers."
      />

      <Card>
        <CardTitle>Synthetic data, used as a design validation</CardTitle>
        <div className="max-w-3xl space-y-3 text-sm leading-relaxed text-muted">
          <p>
            Four seeded tables mirror a real CRM stack: {meta.accounts_total.toLocaleString()} accounts, 717,820
            contacts, 50,000 opportunities, and 122,677 contact-deal role assignments. Win probability is{" "}
            <em>planted</em> by the generator — VP+ presence, technical+business pairing, complementary tech stack,
            segment — and the analysis then has to recover those signals through the full pipeline.
          </p>
          <p>
            That makes every result here a <strong className="text-foreground">design validation</strong>: it proves
            the method recovers known signals at realistic scale, not that these specific coefficients or lifts exist
            in any real market. With production CRM data the signals would be noisier, confounders (rep quality,
            timing, competition) would exist, and temporal dynamics would matter. The architecture transfers; the
            magnitudes do not.
          </p>
        </div>
      </Card>

      <Card>
        <CardTitle>Leakage discipline</CardTitle>
        <p className="max-w-3xl text-sm leading-relaxed text-muted">
          The propensity model uses <strong className="text-foreground">pre-deal features only</strong>:
          firmographics, technographics, and account-level contact composition. In-deal buying-group features — who
          is attached to the opportunity itself — are excluded from the model because they are derived from the same
          opportunities whose outcome is the target. They appear only in the descriptive buying-group analysis. This
          is the single most common way B2B propensity models cheat, and the repo&apos;s tests pin it down.
        </p>
      </Card>

      <Card>
        <CardTitle>Uncertainty: bootstrap everywhere</CardTitle>
        <p className="max-w-3xl text-sm leading-relaxed text-muted">
          Every headline metric carries a{" "}
          <Term def="1,000 resamples of the held-out test set; the 2.5th and 97.5th percentiles of the metric across resamples form the interval.">
            bootstrap 95% confidence interval
          </Term>
          , computed identically for the full model and every baseline so the ladder is a like-for-like comparison.
          That is how the app can say honestly that the full model&apos;s AUC ({model.metrics.auc.toFixed(3)}) is not
          cleanly separable from the firmographic-only baseline — the intervals overlap, and hiding that would
          overstate the contact features.
        </p>
      </Card>

      <Card>
        <CardTitle>Choices worth defending</CardTitle>
        <ul className="max-w-3xl list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-muted">
          <li>
            <strong className="text-foreground">Logistic regression over random forest</strong> — the RF benchmark
            showed no AUC improvement in notebook 03, and coefficients translate directly into scoring rules in
            marketing tools.
          </li>
          <li>
            <strong className="text-foreground">Completeness as four named dimensions</strong> — role coverage,
            seniority mix, function diversity, technical+business pairing — so a low score is immediately a to-do
            list, not just a number.
          </li>
          <li>
            <strong className="text-foreground">The 2.2× stated as association</strong> — the causal version requires
            randomized enrichment, which is the natural next experiment and is framed that way rather than claimed.
          </li>
          <li>
            <strong className="text-foreground">Thresholds (median propensity, completeness 50)</strong> — operating
            choices for the demo, not discovered constants; production teams would tune them to capacity.
          </li>
        </ul>
      </Card>

      <Card>
        <CardTitle>Reproduce it</CardTitle>
        <p className="max-w-3xl text-sm leading-relaxed text-muted">
          The dataset ships in the{" "}
          <a href="https://github.com/pfelix828/lead-scoring" className="underline decoration-dotted underline-offset-2 hover:text-foreground">
            repo
          </a>{" "}
          as seeded parquet; <code className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-xs">scripts/export_web.py</code>{" "}
          recomputes every number on this site from it through the same code paths the analysis used (
          <code className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-xs">src/model.py</code>,{" "}
          <code className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-xs">src/buying_groups.py</code>), and prints
          a parity line against the published figures. Notebooks 01-04 walk the full analysis; 61 pytest tests cover
          the feature, model, and buying-group layers.
        </p>
      </Card>
    </div>
  );
}
