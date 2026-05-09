'use client';

import { useState, useRef } from 'react';
import styles from './ImageUploader.module.css';

interface PredictionResult {
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
  error?: string;
  preprocessedImage?: string;
}

export default function ImageUploader() {
  const [image, setImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [agentQuestion, setAgentQuestion] = useState('');
  const [agentAnswer, setAgentAnswer] = useState<string | null>(null);
  const [agentLoading, setAgentLoading] = useState(false);
  const [agentError, setAgentError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processFile = (file: File) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }

    setFileName(file.name);
    setImageFile(file);
    setError(null);
    setResults(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      setImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    processFile(file);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const handleUploadAreaClick = () => {
    fileInputRef.current?.click();
  };

  const handleAnalyze = async () => {
    if (!image || !imageFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('top_k', '5');

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new Error(text || 'Failed to get predictions');
      }

      const data = await response.json();
      setResults(data);
      setAgentAnswer(null);
      setAgentError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  const handleAskAgent = async () => {
    const question = agentQuestion.trim();
    if (!question) {
      setAgentError('Ask a question first');
      return;
    }

    setAgentLoading(true);
    setAgentError(null);
    setAgentAnswer(null);

    try {
      const context = results?.top_k?.length
        ? {
            top_k: results.top_k,
            fusion_weights: results.fusion_weights,
            explanations: results.explanations
              ? {
                  target_index: results.explanations.target_index,
                  target_label: results.explanations.target_label,
                  cams: {
                    resnet: results.explanations.cams.resnet?.overlay_png
                      ? { overlay_png: results.explanations.cams.resnet.overlay_png }
                      : undefined,
                    efficientnet: results.explanations.cams.efficientnet?.overlay_png
                      ? { overlay_png: results.explanations.cams.efficientnet.overlay_png }
                      : undefined,
                    shufflenet: results.explanations.cams.shufflenet?.overlay_png
                      ? { overlay_png: results.explanations.cams.shufflenet.overlay_png }
                      : undefined,
                  },
                }
              : undefined,
          }
        : null;

      const system =
        'You are a helpful assistant for a retinal disease classification demo. ' +
        'Explain results in plain language for users. ' +
        'Do not diagnose. If uncertain, say so. ' +
        'If you suggest next steps, keep them general and safety-focused. ' +
        'When Grad-CAM overlays are provided in context, you MUST use the available tools to interpret them when the user asks about Grad-CAM/heatmaps/attention or differences between models.';

      const topDisease =
        results?.top_k?.[0]?.label ?? results?.explanations?.target_label ?? null;

      const gradcamImages = results?.explanations?.cams
        ? {
            resnet: results.explanations.cams.resnet?.overlay_png,
            efficientnet: results.explanations.cams.efficientnet?.overlay_png,
            shufflenet: results.explanations.cams.shufflenet?.overlay_png,
          }
        : null;

      const toolHint =
        topDisease && gradcamImages
          ? `\n\nIf you need to interpret the Grad-CAM overlays, call the tool \`gradcam_interpretation\` with:\n` +
            `- top_disease: ${JSON.stringify(topDisease)}\n` +
            `- gradcam_images: {"resnet": <use context.explanations.cams.resnet.overlay_png>, "efficientnet": <use context.explanations.cams.efficientnet.overlay_png>, "shufflenet": <use context.explanations.cams.shufflenet.overlay_png>}\n`
          : '';

      const userMessage = context
        ? `User question: ${question}${toolHint}\n\nModel output context (JSON):\n${JSON.stringify(
            context,
            null,
            2
          )}`
        : question;

      const res = await fetch('/api/agent', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: userMessage },
          ],
        }),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(text || 'Agent request failed');
      }

      const data = (await res.json()) as { message?: string };
      if (!data?.message) {
        throw new Error('Agent returned no message');
      }
      setAgentAnswer(data.message);
    } catch (e) {
      setAgentError(e instanceof Error ? e.message : 'Agent error');
    } finally {
      setAgentLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setImageFile(null);
    setFileName('');
    setResults(null);
    setError(null);
    setAgentQuestion('');
    setAgentAnswer(null);
    setAgentError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className={`${styles.container} dark:${styles.containerDark}`}>
      <div className={styles.maxWidth}>
        {/* Header */}
        <div className={styles.header}>
          <h1 className={`${styles.title} dark:${styles.titleDark}`}>
            Retinal Disease Classification
          </h1>
          <p className={`${styles.subtitle} dark:${styles.subtitleDark}`}>
            Upload a fundus image to analyze for ocular diseases
          </p>
        </div>

        <div className={`${results ? styles.gridLayoutWithResults : styles.gridLayout}`}>
          {/* Image Display Area */}
          <div className={styles.mainArea}>
            {image ? (
              <>
                {/* Original Image Card */}
                <div className={`${styles.card} dark:${styles.cardDark}`}>
                  <div className={styles.imagDisplay}>
                    <div className={styles.transformedImageContainer}>
                      <p className={`${styles.transformedLabel} dark:${styles.transformedLabelDark}`}>
                        Original Image
                      </p>
                      <img
                        src={image}
                        alt="Original uploaded image"
                        className={`${styles.canvas} dark:${styles.canvasDark}`}
                        style={{ maxWidth: '100%', height: 'auto' }}
                      />
                    </div>
                  </div>
                </div>

                {/* Heatmap Visualization Card */}
                {results && (
                  <div className={`${styles.card} dark:${styles.cardDark}`}>
                    <div className={styles.imagDisplay}>
                      <div className={styles.transformedImageContainer}>
                        <p className={`${styles.transformedLabel} dark:${styles.transformedLabelDark}`}>
                          Grad-CAM Explanations
                        </p>
                        {results.explanations?.cams ? (
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '1rem' }}>
                            <div>
                              <p className={`${styles.heatmapText} dark:${styles.heatmapTextDark}`}>ResNet</p>
                              {results.explanations.cams.resnet?.overlay_png ? (
                                <img
                                  src={results.explanations.cams.resnet.overlay_png}
                                  alt="ResNet Grad-CAM"
                                  style={{ width: '100%', height: 'auto', borderRadius: '0.5rem' }}
                                />
                              ) : (
                                <p className={`${styles.heatmapSubtext} dark:${styles.heatmapSubtextDark}`}>No Grad-CAM returned</p>
                              )}
                            </div>
                            <div>
                              <p className={`${styles.heatmapText} dark:${styles.heatmapTextDark}`}>EfficientNet</p>
                              {results.explanations.cams.efficientnet?.overlay_png ? (
                                <img
                                  src={results.explanations.cams.efficientnet.overlay_png}
                                  alt="EfficientNet Grad-CAM"
                                  style={{ width: '100%', height: 'auto', borderRadius: '0.5rem' }}
                                />
                              ) : (
                                <p className={`${styles.heatmapSubtext} dark:${styles.heatmapSubtextDark}`}>No Grad-CAM returned</p>
                              )}
                            </div>
                            <div>
                              <p className={`${styles.heatmapText} dark:${styles.heatmapTextDark}`}>ShuffleNet</p>
                              {results.explanations.cams.shufflenet?.overlay_png ? (
                                <img
                                  src={results.explanations.cams.shufflenet.overlay_png}
                                  alt="ShuffleNet Grad-CAM"
                                  style={{ width: '100%', height: 'auto', borderRadius: '0.5rem' }}
                                />
                              ) : (
                                <p className={`${styles.heatmapSubtext} dark:${styles.heatmapSubtextDark}`}>No Grad-CAM returned</p>
                              )}
                            </div>
                          </div>
                        ) : (
                          <div className={`${styles.heatmapPlaceholder} dark:${styles.heatmapPlaceholderDark}`}>
                            <p className={`${styles.heatmapSubtext} dark:${styles.heatmapSubtextDark}`}>
                              No Grad-CAM data returned
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className={`${styles.card} dark:${styles.cardDark}`}>
                <div
                  className={`${styles.uploadArea} dark:${styles.uploadAreaDark}`}
                  onClick={handleUploadAreaClick}
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  style={{ cursor: 'pointer' }}
                >
                  {/* Hidden file input - no visible button */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                  />
                  <svg
                    className={`${styles.uploadIcon} dark:${styles.uploadIconDark}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                  <p className={`${styles.uploadText} dark:${styles.uploadTextDark}`}>
                    Click to upload or drag image
                  </p>
                  <p className={`${styles.uploadSubtext} dark:${styles.uploadSubtextDark}`}>
                    PNG, JPG up to 10MB
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar - Controls when image loaded */}
          <div className={styles.sidebar}>
            {image && (
              <>
                {/* Controls when image is loaded */}
                <div className={`${styles.sidebarCard} dark:${styles.sidebarCardDark}`}>
                  <p className={`${styles.fileName} dark:${styles.fileNameDark}`}>
                    {fileName}
                  </p>

                  <button
                    onClick={handleAnalyze}
                    disabled={!image || loading}
                    className={styles.analyzeButton}
                  >
                    {loading ? (
                      <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                        <span style={{ animation: 'spin 1s linear infinite' }}>⚙</span> <span style={{ fontSize: '0.75rem' }}>Analyzing</span>
                      </span>
                    ) : (
                      'Analyze'
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={handleReset}
                    className={`${styles.resetButton} dark:${styles.resetButtonDark}`}
                  >
                    Reset
                  </button>
                </div>

                {/* Error Display */}
                {error && (
                  <div className={`${styles.errorCard} dark:${styles.errorCardDark}`}>
                    <p className={styles.errorTitle}>Error</p>
                    <p className={styles.errorMessage}>{error}</p>
                  </div>
                )}

                {/* Predictions */}
                {results && (
                  <div className={`${styles.predictionsCard} dark:${styles.predictionsCardDark}`}>
                    <h3 className={`${styles.predictionsTitle} dark:${styles.predictionsTitleDark}`}>
                      Top Predictions
                    </h3>
                    <div className={styles.predictionsList}>
                      {results.top_k.map((pred, index) => (
                        <div key={`${pred.label}-${pred.index ?? index}`} className={styles.predictionItem}>
                          <div className={styles.predictionHeader}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: 0 }}>
                              <span className={`${styles.predictionRank} dark:${styles.predictionRankDark}`}>
                                {index + 1}
                              </span>
                              <span className={`${styles.predictionName} dark:${styles.predictionNameDark}`}>
                                {pred.label}
                              </span>
                            </div>
                            <span className={`${styles.predictionConfidence} dark:${styles.predictionConfidenceDark}`}>
                              {(pred.probability * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className={`${styles.progressBar} dark:${styles.progressBarDark}`}>
                            <div
                              className={styles.progressFill}
                              style={{
                                width: `${Math.min(pred.probability * 100, 100)}%`,
                                backgroundColor:
                                  index === 0 ? 'rgb(34, 197, 94)' :
                                  index === 1 ? 'rgb(59, 130, 246)' :
                                  index === 2 ? 'rgb(202, 138, 4)' :
                                  index === 3 ? 'rgb(234, 88, 12)' :
                                  'rgb(239, 68, 68)'
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Ask The Agent */}
                <div className={`${styles.diseaseSummaryCard} dark:${styles.diseaseSummaryCardDark}`}>
                  <h3 className={`${styles.diseaseSummaryTitle} dark:${styles.diseaseSummaryTitleDark}`}>
                    Ask The Agent
                  </h3>
                  <p className={`${styles.diseaseSummarySubtitle} dark:${styles.diseaseSummarySubtitleDark}`}>
                    Ask questions about the predictions and Grad-CAM heatmaps
                  </p>

                  <div className={styles.agentPanel}>
                    <div className={styles.agentRow}>
                      <textarea
                        value={agentQuestion}
                        onChange={(e) => setAgentQuestion(e.target.value)}
                        placeholder={
                          results
                            ? 'e.g., What does the top prediction mean?'
                            : 'Upload an image and run Analyze, then ask a question.'
                        }
                        className={`${styles.agentInput} dark:${styles.agentInputDark}`}
                        disabled={agentLoading}
                      />
                      <button
                        type="button"
                        onClick={handleAskAgent}
                        disabled={agentLoading}
                        className={styles.agentButton}
                        title="Send to agent"
                      >
                        {agentLoading ? 'Asking…' : 'Ask'}
                      </button>
                    </div>

                    {agentError && (
                      <div className={`${styles.errorCard} dark:${styles.errorCardDark}`}>
                        <p className={styles.errorTitle}>Agent Error</p>
                        <p className={styles.errorMessage}>{agentError}</p>
                      </div>
                    )}

                    {agentAnswer && (
                      <div className={`${styles.agentAnswer} dark:${styles.agentAnswerDark}`}>
                        {agentAnswer}
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
