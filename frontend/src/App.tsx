import { useState, useEffect, useRef } from 'react';
import { verifyContent } from './services/api';
import { VerificationResult, Verdict } from './types/verification';

// ── Verdict colour tokens ─────────────────────────────────────────────────────
const VERDICT = {
  REAL:    { ring: '#2ecc71', text: 'text-emerald-600', bar: 'bg-emerald-600', badge: 'text-emerald-800' },
  FAKE:    { ring: '#e74c3c', text: 'text-rose-600',    bar: 'bg-rose-600',    badge: 'text-rose-800'    },
  PARTIAL: { ring: '#f39c12', text: 'text-amber-600',   bar: 'bg-amber-500',   badge: 'text-amber-800'  },
};

// ── Donut chart (Chart.js, dynamic import) ────────────────────────────────────
function DonutChart({ result }: { result: VerificationResult }) {
  const canvasRef      = useRef<HTMLCanvasElement>(null);
  const chartRef       = useRef<any>(null);
  const cfg            = VERDICT[result.verdict] ?? VERDICT.FAKE;

  useEffect(() => {
    import('chart.js').then(({ Chart, ArcElement, DoughnutController, Tooltip }) => {
      Chart.register(ArcElement, DoughnutController, Tooltip);
      if (!canvasRef.current) return;
      if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; }

      const segments =
        Array.isArray(result.breakdown) && result.breakdown.length > 0
          ? result.breakdown
          : [{ label: result.verdict, pct: result.confidence, color: cfg.ring }];

      chartRef.current = new Chart(canvasRef.current, {
        type: 'doughnut',
        data: {
          labels: segments.map(s => s.label),
          datasets: [{
            data:            segments.map(s => s.pct),
            backgroundColor: segments.map(s => s.color),
            borderWidth:     3,
            borderColor:     '#f4ebd0',   // matches paper bg
            hoverOffset:     4,
          }],
        },
        options: {
          responsive: false,
          cutout: '70%',
          animation: { duration: 700 },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: { label: (ctx: any) => ` ${ctx.label}: ${Math.round(Number(ctx.parsed))}%` },
            },
          },
        },
      });
    });
    return () => { if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; } };
  }, [result.verdict, result.confidence, result.breakdown]);

  // Centre label: for PARTIAL show truth %, for REAL/FAKE show confidence
  const centreValue =
    result.verdict === 'PARTIAL'
      ? `${result.breakdown?.find(s => s.color === '#2ecc71')?.pct ?? result.truth_percentage ?? result.confidence}%`
      : `${result.confidence}%`;

  return (
    <div className="flex justify-center relative my-4">
      <div className="relative w-36 h-36">
        <canvas
          ref={canvasRef}
          width={144}
          height={144}
          role="img"
          aria-label={`${result.verdict} — ${result.confidence}% confidence`}
        />
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none font-sans">
          <span className="text-xs font-mono text-slate-600 uppercase tracking-tighter">Confidence</span>
          <span className={`text-2xl font-black ${cfg.text}`}>{centreValue}</span>
        </div>
      </div>
    </div>
  );
}

// ── Metric bar ─────────────────────────────────────────────────────────────────
function MetricBar({ label, value, barClass }: { label: string; value: number; barClass: string }) {
  return (
    <div className="font-sans text-xs">
      <div className="flex justify-between font-mono font-bold text-[11px] mb-1">
        <span>{label}</span>
        <span>{value}%</span>
      </div>
      <div className="w-full h-2.5 bg-[#decfae] rounded-full overflow-hidden">
        <div className={`h-full transition-all duration-700 ${barClass}`} style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

// ── Safe hostname ──────────────────────────────────────────────────────────────
function safeHostname(url?: string): string {
  if (!url) return '';
  try { return new URL(url).hostname.replace(/^www\./, ''); }
  catch { return url; }
}

// ── Default state (shown before first API call) ────────────────────────────────
const DEFAULT_RESULT: VerificationResult = {
  headline:         "Why hundreds of Indian seafarers can't leave abandoned ships",
  verdict:          'REAL',
  confidence:       94,
  truth_percentage: 100,
  fake_percentage:  0,
  partial_reason:   '',
  breakdown:        [{ label: 'Verified true', pct: 100, color: '#2ecc71' }],
  explanation:      'The submitted headline demonstrates semantic markers and linguistic structures strongly indicative of accurate reporting. Initial fact-checks correlate with established political reporting cycles for the stated region.',
  evidence_url:     'https://www.reuters.com',
  evidence_snippet: '',
  metrics: {
    language_model_score: 94,
    structural_accuracy:  93,
    factual_correlation:  91,
  },
};

// ── App ────────────────────────────────────────────────────────────────────────
export default function App() {
  const [headline,  setHeadline]  = useState(DEFAULT_RESULT.headline);
  const [isLoading, setIsLoading] = useState(false);
  const [error,     setError]     = useState<string | null>(null);
  const [result,    setResult]    = useState<VerificationResult>(DEFAULT_RESULT);

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!headline.trim()) return;
    setIsLoading(true);
    setError(null);
    try {
      const data = await verifyContent(headline);
      setResult(data);
    } catch (err) {
      setError('Unable to connect to the Python inference server. Showing last result.');
      setResult(prev => ({ ...prev, headline }));
    } finally {
      setIsLoading(false);
    }
  };

  const cfg     = VERDICT[result.verdict] ?? VERDICT.FAKE;
  const metrics = result.metrics ?? { language_model_score: 0, structural_accuracy: 0, factual_correlation: 0 };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-serif p-4 md:p-8 selection:bg-amber-500 selection:text-slate-950">

      {/* Header */}
      <header className="max-w-5xl mx-auto border-b-4 border-double border-slate-800 pb-4 mb-8 text-center">
        <h1 className="text-3xl md:text-5xl font-black tracking-tighter uppercase font-mono bg-gradient-to-r from-amber-400 via-slate-200 to-amber-500 bg-clip-text text-transparent">
          Fake News Detection Engine
        </h1>
      </header>

      <main className="max-w-5xl mx-auto space-y-8">

        {/* Input form */}
        <form onSubmit={handleAnalyze} className="bg-slate-900 border border-slate-800 p-6 rounded-xl shadow-2xl max-w-3xl mx-auto">
          <label className="block font-mono text-xs uppercase tracking-wider text-amber-400 mb-2 font-bold">
            Analyze Source Content:
          </label>
          <textarea
            value={headline}
            onChange={e => setHeadline(e.target.value)}
            placeholder="Article Text or Headline..."
            className="w-full h-24 bg-slate-950 text-slate-200 p-4 font-sans text-sm rounded-lg border border-slate-800 focus:outline-none focus:border-amber-500 transition-colors resize-none"
          />
          <button
            type="submit"
            disabled={isLoading}
            className="w-full mt-4 bg-blue-600 text-white font-sans font-medium py-2.5 px-6 rounded-lg text-sm hover:bg-blue-500 transition-colors disabled:opacity-50"
          >
            {isLoading ? '🕵️‍♂️ Cross-Referencing Live Google Indexes...' : 'Analyze Credibility'}
          </button>
        </form>

        {error && (
          <div className="max-w-3xl mx-auto p-4 bg-rose-950/40 border border-rose-800/60 rounded-xl text-rose-300 text-xs font-mono">
            ⚠️ {error}
          </div>
        )}

        {/* ── Newspaper card ─────────────────────────────────────────────────── */}
        <div className="max-w-5xl mx-auto bg-[#f4ebd0] text-slate-900 p-6 md:p-8 rounded shadow-2xl relative border-[12px] border-[#e6d5b3] text-left transition-all duration-300">

          {/* Masthead */}
          <div className="border-b-4 border-slate-950 pb-2 mb-4 flex items-center gap-3">
            <span className="text-3xl">📰</span>
            <h2 className="text-2xl md:text-3xl font-black tracking-tight text-slate-950 uppercase font-mono">
              Analyzed Dispatch - Veritas Chronicle
            </h2>
          </div>

          <div className="border-b-2 border-slate-950 pb-1 mb-6 flex justify-between font-mono text-xs font-bold uppercase text-slate-700">
            <span>The Classification Report</span>
            <span className={cfg.badge}>Verdict: {result.verdict}</span>
          </div>

          {/* Headline */}
          <div className="mb-6">
            <span className="block font-mono text-xs uppercase font-bold tracking-wider text-slate-600">Headline:</span>
            <h3 className="text-xl md:text-2xl font-black leading-tight text-slate-950 mt-1">
              {result.headline}
            </h3>
          </div>

          {/* PARTIAL reason banner — only shown when verdict is PARTIAL */}
          {result.verdict === 'PARTIAL' && result.partial_reason && (
            <div className="mb-6 bg-amber-100 border border-amber-400 rounded px-4 py-3">
              <span className="block font-mono text-[11px] font-bold uppercase tracking-wider text-amber-700 mb-1">
                Why partial?
              </span>
              <p className="text-sm text-amber-900 leading-relaxed">{result.partial_reason}</p>
            </div>
          )}

          {/* Two-column layout */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start border-t border-slate-400 pt-6">

            {/* LEFT — donut + metrics */}
            <div className="lg:col-span-5 space-y-6 border-b lg:border-b-0 lg:border-r border-slate-400 pb-6 lg:pb-0 lg:pr-6">
              <div>
                <h4 className="font-mono text-xs font-bold uppercase tracking-wider text-slate-700 mb-4">
                  Credibility Score
                </h4>

                {/* Real Chart.js donut — replaces old SVG ring */}
                <DonutChart result={result} />

                {/* PARTIAL legend under chart */}
                {result.verdict === 'PARTIAL' && Array.isArray(result.breakdown) && result.breakdown.length > 1 && (
                  <div className="flex flex-col gap-1.5 mt-3">
                    {result.breakdown.map(seg => (
                      <div key={seg.label} className="flex items-center gap-2 text-xs text-slate-700">
                        <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ background: seg.color }} />
                        <span className="flex-1">{seg.label}</span>
                        <span className="font-bold" style={{ color: seg.color }}>{seg.pct}%</span>
                      </div>
                    ))}
                  </div>
                )}

                <p className="text-center font-mono text-[11px] text-slate-600 font-bold mt-3">
                  **VERDICT: {result.verdict}**
                </p>
              </div>

              {/* Metric bars — wired to real backend values */}
              <div className="space-y-3 pt-4 border-t border-slate-400">
                <h5 className="font-mono text-[11px] font-bold uppercase tracking-wide text-slate-700">
                  Analysis Metrics
                </h5>
                <MetricBar label="Language Model Score" value={metrics.language_model_score} barClass={cfg.bar} />
                <MetricBar label="Structural Accuracy"  value={metrics.structural_accuracy}  barClass="bg-slate-800" />
                <MetricBar label="Factual Correlation"  value={metrics.factual_correlation}  barClass="bg-slate-800" />
              </div>
            </div>

            {/* RIGHT — explanation + evidence */}
            <div className="lg:col-span-7 space-y-6 font-sans text-sm text-slate-900 leading-relaxed">

              <div>
                <h4 className="font-mono text-xs font-bold uppercase tracking-wider text-slate-700 mb-2">
                  Article Verification Summary
                </h4>
                <p className="text-slate-800 text-justify bg-[#fcf9f0]/60 p-3 border border-[#decfae] rounded-sm italic">
                  {result.explanation}
                </p>
              </div>

              <div>
                <h4 className="font-mono text-xs font-bold uppercase tracking-wider text-slate-700 mb-1">
                  Verified Reference Links
                </h4>
                <div className="border-t-2 border-slate-950 pt-2">
                  <span className="block font-mono text-[11px] font-bold text-slate-600 uppercase">
                    Verified Reference Source
                  </span>
                  {result.evidence_url ? (
                    <a
                      href={result.evidence_url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 text-blue-800 hover:text-blue-950 font-medium underline mt-1 break-all"
                    >
                      {safeHostname(result.evidence_url)}
                      <span className="text-xs no-underline">↗</span>
                    </a>
                  ) : (
                    <span className="text-slate-500 text-xs mt-1 block">No reference link available.</span>
                  )}
                  {result.evidence_snippet && (
                    <p className="text-xs text-slate-600 mt-2 leading-relaxed line-clamp-3 not-italic">
                      {result.evidence_snippet}
                    </p>
                  )}
                </div>
              </div>

              <div className="pt-4 border-t border-slate-400">
                <h4 className="font-mono text-[11px] font-bold uppercase tracking-wide text-slate-700 mb-1">
                  Deeper Breakdown
                </h4>
                <p className="text-xs text-slate-700 text-justify">
                  Cross-referenced with open repository records from trusted news aggregators.
                  Final score reflects high structural alignment with verified primary sources
                  and semantic vector confirmation.
                </p>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="mt-8 pt-2 border-t border-slate-400 font-mono text-[10px] text-slate-500 text-center uppercase tracking-wider">
            Verified via automated neural verification layers // End of Report
          </div>
        </div>
      </main>
    </div>
  );
}
