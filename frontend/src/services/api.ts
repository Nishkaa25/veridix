import { VerificationResult } from '../types/verification';

export const verifyContent = async (text: string): Promise<VerificationResult> => {
  const response = await fetch('http://127.0.0.1:8000/api/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // backend now expects { text } — body field optional if user pastes full article
    body: JSON.stringify({ title: text, body: '' }),
  });

  if (!response.ok) {
    throw new Error('Backend engine connection failed');
  }

  const data = await response.json();

  return {
    headline:         text,
    verdict:          data.verdict,           // 'REAL' | 'FAKE' | 'PARTIAL'
    confidence:       data.confidence,
    truth_percentage: data.truth_percentage,  // 0-100
    fake_percentage:  data.fake_percentage,   // 0-100
    partial_reason:   data.partial_reason,    // empty string if not PARTIAL
    breakdown:        data.breakdown,         // [{label, pct, color}]
    explanation:      data.explanation,
    evidence_url:     data.evidence_url,
    evidence_snippet: data.evidence_snippet,
    metrics:          data.metrics,           // {language_model_score, structural_accuracy, factual_correlation}
  };
};