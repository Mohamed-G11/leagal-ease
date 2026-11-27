// Citations of an answer, and the [n] markers that point at them.

export interface Citation {
  chunk_id: string;
  doc_title: string;
  page: number;
  quote: string;
  /** The [n] the answer text uses for this source. Absent on older answers. */
  number?: number | null;
}

// A run of markers: [3], [1][2], [1, 2], or the full-width 【3】 some models write.
// The space before the run is captured, so that it goes away when no marker of the run is kept.
const MARKERS = /([ \t]?)((?:[[【]\d+(?:\s*,\s*\d+)*[\]】])+)/g;

/** Keep only the markers that point at a listed source, so every [n] left in the text can be looked up. */
export function cleanMarkers(answer: string, citations: Citation[]): string {
  const known = new Set(citations.map(c => c.number));
  return answer.replace(MARKERS, (_match, space: string, run: string) => {
    const kept = (run.match(/\d+/g) ?? []).map(Number).filter(n => known.has(n));
    return kept.length ? space + kept.map(n => `[${n}]`).join('') : '';
  });
}

/** Numbered sources in marker order, then those without a number. */
export function sortCitations(citations: Citation[]): Citation[] {
  return [...citations].sort((a, b) => (a.number ?? Infinity) - (b.number ?? Infinity));
}
