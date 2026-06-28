import React, { useState } from 'react';

interface SourceInputProps {
  onVerify: (text: string) => void;
  isLoading: boolean;
}

export const SourceInput: React.FC<SourceInputProps> = ({ onVerify, isLoading }) => {
  const [text, setText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    onVerify(text);
  };

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-100">
      <h2 className="text-lg font-semibold text-slate-800 mb-4">Analyze Source Content</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-600 mb-1">
            Article Text or Headline
          </label>
          <textarea
            rows={5}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste the news headline or complete article text here to initiate deep web-verification..."
            required
            className="w-full rounded-lg border border-slate-200 p-3 text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
          />
        </div>
        <button
          type="submit"
          disabled={isLoading}
          className={`w-full py-3 px-4 rounded-lg font-semibold text-white transition-all ${
            isLoading 
              ? 'bg-blue-400 cursor-not-allowed animate-pulse' 
              : 'bg-blue-600 hover:bg-blue-700 shadow-sm shadow-blue-200'
          }`}
        >
          {isLoading ? '🕵️‍♂️ Cross-Referencing Live Google Indexes...' : 'Analyze Credibility'}
        </button>
      </form>
    </div>
  );
};