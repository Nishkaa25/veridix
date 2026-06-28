export interface BreakdownSegment {
  label: string;
  pct: number;
  color: string;
}

export interface Metrics {
  language_model_score: number;
  structural_accuracy: number;
  factual_correlation: number;
}

export type Verdict = 'REAL' | 'FAKE' | 'PARTIAL';

export interface VerificationResult {
  headline: string;
  verdict: Verdict;
  confidence: number;
  truth_percentage: number;
  fake_percentage: number;
  partial_reason: string;
  breakdown: BreakdownSegment[];
  explanation: string;
  evidence_url: string;
  evidence_snippet: string;
  metrics: Metrics;
}