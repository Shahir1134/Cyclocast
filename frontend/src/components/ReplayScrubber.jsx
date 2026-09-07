import React, { useState, useEffect } from 'react';
import { Play, Pause, SkipBack, SkipForward, Clock, Target, Wind } from 'lucide-react';
import { getCategoryColor } from '../utils/constants';

export default function ReplayScrubber({
  replayData,
  currentStepIndex,
  onStepChange,
}) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed] = useState(1500); // ms per step

  const steps = replayData?.steps || [];
  const totalSteps = steps.length;
  const safeIndex = totalSteps > 0 ? Math.min(Math.max(0, currentStepIndex || 0), totalSteps - 1) : 0;
  const currentStep = steps[safeIndex] || null;

  // Auto-play interval - MUST be called on every render before any conditional returns
  useEffect(() => {
    let interval = null;
    if (isPlaying && totalSteps > 0) {
      interval = setInterval(() => {
        onStepChange((prev) => {
          if (prev >= totalSteps - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, playbackSpeed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, totalSteps, playbackSpeed, onStepChange]);

  // Early return for loading or empty state (placed AFTER all hooks)
  if (!replayData || !steps || steps.length === 0 || !currentStep) {
    return (
      <div className="w-full bg-gray-950/90 border border-purple-900/40 rounded-xl p-4 flex items-center justify-center gap-2 text-xs text-purple-300">
        <span className="w-3.5 h-3.5 rounded-full border-2 border-purple-400 border-t-transparent animate-spin"></span>
        <span>Loading historical benchmark ground-truth verification data...</span>
      </div>
    );
  }

  const handleSliderChange = (e) => {
    onStepChange(parseInt(e.target.value, 10));
  };

  const handlePrev = () => {
    onStepChange((prev) => Math.max(0, prev - 1));
  };

  const handleNext = () => {
    onStepChange((prev) => Math.min(steps.length - 1, prev + 1));
  };

  let formattedDate = 'Historical Step';
  try {
    if (currentStep?.timestamp) {
      formattedDate = new Date(currentStep.timestamp).toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZone: 'UTC',
      }) + ' UTC';
    }
  } catch {
    formattedDate = `Step ${safeIndex + 1}`;
  }

  const avgTrackError = replayData?.mean_track_error_km != null ? `${replayData.mean_track_error_km} km` : '--';
  const avgIntError = replayData?.mean_intensity_error_kt != null ? `${replayData.mean_intensity_error_kt} kt` : '--';

  return (
    <div className="w-full bg-gray-950/90 border border-purple-900/40 rounded-xl p-3 flex flex-col gap-2.5 shadow-lg">
      {/* Top row: controls + timestamp + live error badges */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        {/* Playback Controls */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={handlePrev}
            disabled={safeIndex === 0}
            className="p-1.5 rounded-lg bg-gray-900 border border-gray-800 text-gray-300 hover:text-white disabled:opacity-30 transition-all"
            title="Previous step"
          >
            <SkipBack className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-3 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-semibold flex items-center gap-1.5 text-xs shadow-md shadow-purple-600/30 transition-all cursor-pointer"
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isPlaying ? 'Pause' : 'Play'}</span>
          </button>

          <button
            onClick={handleNext}
            disabled={safeIndex === steps.length - 1}
            className="p-1.5 rounded-lg bg-gray-900 border border-gray-800 text-gray-300 hover:text-white disabled:opacity-30 transition-all"
            title="Next step"
          >
            <SkipForward className="w-3.5 h-3.5" />
          </button>

          <div className="ml-2 flex items-center gap-1 text-[11px] text-gray-400 font-mono">
            <Clock className="w-3 h-3 text-purple-400" />
            <span>{formattedDate}</span>
          </div>
        </div>

        {/* Real-time Error comparison readout */}
        <div className="flex items-center gap-2">
          <div className="bg-gray-900 border border-gray-800 px-2.5 py-1 rounded-lg flex items-center gap-2 text-xs">
            <div className="flex items-center gap-1 text-gray-400">
              <Target className="w-3.5 h-3.5 text-cyan-400" />
              <span>Step Error:</span>
            </div>
            <span className="font-bold text-cyan-300">
              {currentStep.track_error_km ?? 0} km
            </span>
          </div>

          <div className="bg-gray-900 border border-gray-800 px-2.5 py-1 rounded-lg flex items-center gap-2 text-xs">
            <div className="flex items-center gap-1 text-gray-400">
              <Wind className="w-3.5 h-3.5 text-amber-400" />
              <span>Intensity Error:</span>
            </div>
            <span className="font-bold text-amber-300">
              {currentStep.intensity_error_kt ?? 0} kt
            </span>
          </div>

          <div className="hidden sm:flex text-[10px] text-gray-400 font-medium px-2 py-1 rounded bg-purple-950/40 border border-purple-800/30">
            Avg Track: {avgTrackError} • Avg Int: {avgIntError}
          </div>
        </div>
      </div>

      {/* Timeline Slider */}
      <div className="flex items-center gap-3">
        <span className="text-[10px] text-gray-400 font-mono">
          Step {safeIndex + 1}/{steps.length}
        </span>
        <input
          type="range"
          min="0"
          max={Math.max(0, steps.length - 1)}
          value={safeIndex}
          onChange={handleSliderChange}
          className="w-full h-1.5 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
        />
      </div>

      {/* Side-by-Side Step Stats Comparison */}
      <div className="grid grid-cols-2 gap-2 text-[11px]">
        {/* Actual Truth */}
        <div className="p-2 rounded-lg bg-emerald-950/20 border border-emerald-800/30 flex justify-between items-center">
          <div>
            <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Ground Truth (IMD Best-Track)</span>
            <div className="font-mono text-gray-200 mt-0.5">
              {currentStep.actual_lat?.toFixed(2)}°N, {currentStep.actual_lon?.toFixed(2)}°E • {Math.round(currentStep.actual_wind_kt || 0)} kt
            </div>
          </div>
          <span
            className="text-[9px] font-bold px-1.5 py-0.5 rounded border"
            style={{
              color: getCategoryColor(currentStep.actual_category),
              borderColor: `${getCategoryColor(currentStep.actual_category)}40`,
            }}
          >
            {currentStep.actual_category || 'Storm'}
          </span>
        </div>

        {/* Model Prediction */}
        <div className="p-2 rounded-lg bg-purple-950/20 border border-purple-800/30 flex justify-between items-center">
          <div>
            <span className="text-[10px] font-bold text-purple-400 uppercase tracking-wider">CycloCast Model Output</span>
            <div className="font-mono text-gray-200 mt-0.5">
              {currentStep.predicted_lat?.toFixed(2)}°N, {currentStep.predicted_lon?.toFixed(2)}°E • {Math.round(currentStep.predicted_wind_kt || 0)} kt
            </div>
          </div>
          <span
            className="text-[9px] font-bold px-1.5 py-0.5 rounded border"
            style={{
              color: getCategoryColor(currentStep.predicted_category),
              borderColor: `${getCategoryColor(currentStep.predicted_category)}40`,
            }}
          >
            {currentStep.predicted_category || 'Storm'}
          </span>
        </div>
      </div>
    </div>
  );
}
