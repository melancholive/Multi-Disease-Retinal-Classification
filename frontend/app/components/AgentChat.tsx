'use client';

import { useRef, useState } from 'react';
import styles from './ImageUploader.module.css';

interface AgentPredictionResult {
  top_k: Array<{
    label: string;
    probability: number;
    index?: number;
  }>;
  fusion_weights?: number[];
  explanations?: {
    target_index: number | null;
    target_label: string | null;
    cams: {
      resnet?: { overlay_png: string };
      efficientnet?: { overlay_png: string };
      shufflenet?: { overlay_png: string };
    };
  };
}

interface AgentExchange {
  id: string;
  question: string;
  answer: string | null;
}

interface AgentChatProps {
  fileName: string;
  results: AgentPredictionResult | null;
}

export default function AgentChat({ fileName, results }: AgentChatProps) {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState<AgentExchange[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionIdRef = useRef<string>(crypto.randomUUID());

  const handleAskAgent = async () => {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      setError('Ask a question first');
      return;
    }

    const exchangeId = crypto.randomUUID();

    setLoading(true);
    setError(null);
    setQuestion('');
    setMessages((currentMessages) => [
      ...currentMessages,
      {
        id: exchangeId,
        question: trimmedQuestion,
        answer: null,
      },
    ]);

    try {
      const topDisease =
        results?.top_k?.[0]?.label ?? results?.explanations?.target_label ?? null;

      const gradcamImages = results?.explanations?.cams
        ? {
            resnet: results.explanations.cams.resnet?.overlay_png,
            efficientnet: results.explanations.cams.efficientnet?.overlay_png,
            shufflenet: results.explanations.cams.shufflenet?.overlay_png,
          }
        : null;

      const predictionContext = results?.top_k?.length
        ? {
            top_k: results.top_k,
            fusion_weights: results.fusion_weights,
            explanations: results.explanations
              ? {
                  target_index: results.explanations.target_index,
                  target_label: results.explanations.target_label,
                }
              : undefined,
          }
        : undefined;

      const res = await fetch('/api/agent', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          message: trimmedQuestion,
          session_id: sessionIdRef.current,
          image_path: fileName || null,
          top_diseases: results?.top_k ?? null,
          top_k: results?.top_k?.length ?? null,
          prediction_context: predictionContext,
          gradcam_images: gradcamImages,
          default_gradcam_model:
            topDisease && gradcamImages ? topDisease.toLowerCase().replace(/\s+/g, '_') : null,
        }),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(text || 'Agent request failed');
      }

      const data = (await res.json()) as { message?: string; answer?: string };
      const agentText = data?.message ?? data?.answer;
      if (!agentText) {
        throw new Error('Agent returned no message');
      }

      setMessages((currentMessages) =>
        currentMessages.map((message) =>
          message.id === exchangeId ? { ...message, answer: agentText } : message
        )
      );
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'Agent error';
      setError(errorMessage);
      setMessages((currentMessages) =>
        currentMessages.map((message) =>
          message.id === exchangeId
            ? { ...message, answer: `Agent error: ${errorMessage}` }
            : message
        )
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`${styles.diseaseSummaryCard} dark:${styles.diseaseSummaryCardDark}`}>
      <h3 className={`${styles.diseaseSummaryTitle} dark:${styles.diseaseSummaryTitleDark}`}>
        Ask The Agent
      </h3>
      <p className={`${styles.diseaseSummarySubtitle} dark:${styles.diseaseSummarySubtitleDark}`}>
        Ask questions about the predictions and Grad-CAM heatmaps
      </p>

      <div className={styles.agentPanel}>
        <div className={`${styles.agentChatBox} dark:${styles.agentChatBoxDark}`}>
          {messages.length > 0 ? (
            <div className={styles.agentTranscript}>
              {messages.map((message) => (
                <div className={styles.agentExchange} key={message.id}>
                  <div className={`${styles.agentQuestion} dark:${styles.agentQuestionDark}`}>
                    {message.question}
                  </div>
                  <div
                    className={`${styles.agentAnswer} dark:${styles.agentAnswerDark} ${
                      message.answer ? '' : styles.agentAnswerPending
                    }`}
                  >
                    {message.answer ?? 'Thinking...'}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className={`${styles.agentEmptyState} dark:${styles.agentEmptyStateDark}`}>
              Ask a question to start the conversation.
            </div>
          )}
        </div>

        {error && (
          <div className={`${styles.errorCard} dark:${styles.errorCardDark}`}>
            <p className={styles.errorTitle}>Agent Error</p>
            <p className={styles.errorMessage}>{error}</p>
          </div>
        )}

        <div className={styles.agentRow}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={
              results
                ? 'e.g., What does the top prediction mean?'
                : 'Upload an image and run Analyze, then ask a question.'
            }
            className={`${styles.agentInput} dark:${styles.agentInputDark}`}
            disabled={loading}
          />
          <button
            type="button"
            onClick={handleAskAgent}
            disabled={loading}
            className={styles.agentButton}
            title="Send to agent"
          >
            {loading ? 'Asking...' : 'Ask'}
          </button>
        </div>
      </div>
    </div>
  );
}
