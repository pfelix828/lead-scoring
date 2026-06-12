import { model } from "@/lib/data";
import { pct } from "@/lib/format";
import { Card, CardTitle, PageHeader, Stat, SyntheticDataNote, Term } from "@/components/ui";
import { BaselinesLadder, CalibrationChart, ImportanceBars, LiftCaptureChart, ScoreHistogram } from "@/components/charts";

export default function ModelPage() {
  const m = model.metrics;
  const c = model.calibration;
  const d1 = model.lift[0];
  const top2capture = model.lift[1]?.cumulative_capture;
  return (
    <div className="space-y-6">
      <PageHeader
        title="Is an AUC of 0.575 any good?"
        subtitle={
          <>
            Honest answer: it&apos;s modest, it&apos;s realistic for pre-deal B2B propensity, and whether it&apos;s
            useful depends on the question you ask of it. This page answers three in order: is the model better than
            cheap alternatives, can you trust its probabilities, and what does it buy you operationally.
          </>
        }
      />

      <Card>
        <CardTitle sub="Test AUC with bootstrap 95% CIs. Each baseline isolates a progressively richer signal; the full model must beat them to earn its 39 features.">
          1 · Better than the cheap alternatives?
        </CardTitle>
        <BaselinesLadder baselines={model.baselines} />
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted">
          A single SQL-sortable feature — senior contact density — already reaches 0.548. Firmographics alone reach
          0.562. The full model&apos;s 0.575 beats both, but its CI overlaps the firmographic baseline, so the honest
          read is: <strong className="text-foreground">most pre-deal signal is firmographic, and contact-composition
          features add marginal ranking lift</strong>. The contact data earns its keep elsewhere — in the buying-group
          analysis, where it describes <em>what to fix</em> rather than predicts who buys.
        </p>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardTitle sub="Predicted vs observed win rate across ten score deciles on the held-out test set.">
            2 · Can you trust the probabilities?
          </CardTitle>
          <CalibrationChart bins={c.bins} />
          <p className="mt-2 text-sm leading-relaxed text-muted">
            Yes — the largest predicted-vs-observed gap is{" "}
            <strong className="text-foreground">{(c.max_gap * 100).toFixed(1)} points</strong>, and the{" "}
            <Term def="Mean squared error of the predicted probabilities; lower is better. The no-skill comparison always predicts the base win rate.">
              Brier score
            </Term>{" "}
            is {c.brier.toFixed(3)} vs {c.brier_no_skill.toFixed(3)} for no-skill. A weak separator with trustworthy
            probabilities is exactly the model you can still set thresholds on.
          </p>
        </Card>

        <Card>
          <CardTitle sub="Score distributions for won (purple) and lost (gray) test accounts.">
            Why separation is hard, visibly
          </CardTitle>
          <ScoreHistogram hist={model.score_hist} />
          <p className="mt-2 text-sm leading-relaxed text-muted">
            The distributions overlap heavily — that <em>is</em> AUC 0.575, drawn instead of quoted. Pre-deal
            firmographic and contact signals are weak on their own; no amount of model complexity conjures separation
            the features don&apos;t carry.
          </p>
        </Card>
      </div>

      <Card>
        <CardTitle sub="Lift over random per decile (bars, D1 = highest scores) and the running share of all wins captured (line).">
          3 · What it buys you operationally
        </CardTitle>
        <LiftCaptureChart lift={model.lift} />
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted">
          The top decile wins at {pct(d1.win_rate, 1)} — {d1.lift.toFixed(1)}× the base rate — and the top two
          deciles already hold {top2capture ? pct(top2capture) : ""} of all wins. For a marketing team allocating
          budget across 100K accounts, concentrating spend up-rank is the entire value of the model, and it survives
          a modest AUC.
        </p>
      </Card>

      <Card>
        <CardTitle sub="Logistic-regression coefficients, top 20 by magnitude. Green pushes toward winning; red away.">
          What drives the score
        </CardTitle>
        <ImportanceBars importance={model.importance} />
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted">
          Interpretable on purpose: coefficients translate directly to scoring rules in tools like Marketo or
          HubSpot. A random-forest benchmark scored no better (AUC 0.567 vs 0.575 in the repo&apos;s notebook 03), so
          the explainable model won. Note the synthetic-data caveat: these weights recover signals the generator
          planted — direction is meaningful, magnitude is not transferable.
        </p>
      </Card>

      <div className="grid gap-4 sm:grid-cols-4">
        <Card><Stat label="Test AUC" value={m.auc.toFixed(3)} hint={`95% CI ${m.auc_ci_lower.toFixed(3)}-${m.auc_ci_upper.toFixed(3)}`} /></Card>
        <Card><Stat label="CV AUC" value={m.cv_auc.toFixed(3)} hint="5-fold cross-validation on the training set." /></Card>
        <Card><Stat label="Precision @ top 10%" value={pct(m.precision_at_10pct)} hint={`95% CI ${pct(m.precision_at_10pct_ci_lower)}-${pct(m.precision_at_10pct_ci_upper)}`} /></Card>
        <Card><Stat label="Log loss" value={m.log_loss.toFixed(3)} hint="Lower is better; complements Brier on probability quality." /></Card>
      </div>

      <SyntheticDataNote />
    </div>
  );
}
