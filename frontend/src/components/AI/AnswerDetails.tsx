// What is shown under an answer: how sure the system is, and where the answer comes from.
import { sortCitations, type Citation } from './citations';

interface Props {
  status?: 'answered' | 'abstained';
  /** 0 to 1. The current pipeline does not compute it yet. */
  confidence?: number | null;
  citations?: Citation[];
}

function Confidence({ status, confidence }: Pick<Props, 'status' | 'confidence'>) {
  if (status === 'abstained') {
    return <div className="aip-conf abstained">Not found in the documents</div>;
  }
  if (typeof confidence !== 'number') {
    return <div className="aip-conf unknown">Confidence: not provided</div>;
  }
  const percent = Math.round(Math.min(Math.max(confidence, 0), 1) * 100);
  const level = percent >= 70 ? 'high' : percent >= 40 ? 'medium' : 'low';
  return (
    <div className={`aip-conf ${level}`}>
      <span className="aip-conf-bar"><span style={{ width: `${percent}%` }} /></span>
      Confidence: {level} ({percent}%)
    </div>
  );
}

export default function AnswerDetails({ status, confidence, citations = [] }: Props) {
  return (
    <>
      <Confidence status={status} confidence={confidence} />
      {citations.length > 0 && (
        <div className="aip-sources">
          {sortCitations(citations).map(c => (
            <details key={c.chunk_id} className="aip-src-item">
              <summary title={c.quote}>
                {c.number ? <b>[{c.number}]</b> : <span className="aip-src-dot" />}
                <span className="aip-src-title">{c.doc_title}</span>
                <span className="aip-src-page">p. {c.page}</span>
              </summary>
              <blockquote>{c.quote}</blockquote>
            </details>
          ))}
        </div>
      )}
    </>
  );
}
