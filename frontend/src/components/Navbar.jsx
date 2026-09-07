import React from 'react';
import { Compass, Radio, History, Eye, ShieldAlert, Sparkles, Activity } from 'lucide-react';

export default function Navbar({
  isReplayMode,
  setIsReplayMode,
  showSatellite,
  setShowSatellite,
  selectedBasin,
  setSelectedBasin,
  onTriggerLiveForecast,
  isLoadingLive,
}) {
  return (
    <header className="h-14 bg-gray-950 border-b border-gray-800 px-4 flex items-center justify-between select-none z-30 flex-shrink-0">
      {/* Brand & Subtitle */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-blue-600 via-indigo-600 to-cyan-400 flex items-center justify-center shadow-lg shadow-blue-500/20">
          <Compass className="w-5 h-5 text-white animate-spin-slow" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-bold tracking-wider text-white flex items-center gap-1.5">
              CYCLO<span className="text-cyan-400">CAST</span>
            </h1>
            <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
              SIH 26070 • MoES / IMD
            </span>
          </div>
          <p className="text-[11px] text-gray-400 leading-none">
            Multimodal AI Cyclone Forecasting (GraphCast + Meteosat-8 ResNet-18)
          </p>
        </div>
      </div>

      {/* Center Controls */}
      <div className="flex items-center gap-3">
        {/* Mode Selector */}
        <div className="bg-gray-900 border border-gray-800 p-1 rounded-lg flex items-center gap-1 text-xs">
          <button
            onClick={() => setIsReplayMode(false)}
            className={`px-3 py-1 rounded-md font-medium flex items-center gap-1.5 transition-all ${
              !isReplayMode
                ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Radio className="w-3.5 h-3.5" />
            Live Forecast
          </button>
          <button
            onClick={() => setIsReplayMode(true)}
            className={`px-3 py-1 rounded-md font-medium flex items-center gap-1.5 transition-all ${
              isReplayMode
                ? 'bg-purple-600 text-white shadow-md shadow-purple-600/30'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <History className="w-3.5 h-3.5" />
            Replay Mode
          </button>
        </div>

        {/* Satellite Overlay Toggle */}
        <button
          onClick={() => setShowSatellite(!showSatellite)}
          className={`px-3 py-1.5 rounded-lg border text-xs font-medium flex items-center gap-1.5 transition-all ${
            showSatellite
              ? 'bg-cyan-950/70 border-cyan-500 text-cyan-300 shadow-sm shadow-cyan-500/20'
              : 'bg-gray-900 border-gray-800 text-gray-400 hover:text-white'
          }`}
          title="Toggle live Meteosat-8 IR calibrated cloud layer"
        >
          <Eye className="w-3.5 h-3.5" />
          Satellite IR Overlay
        </button>

        {/* Live ML Refresh Button */}
        {!isReplayMode && (
          <button
            onClick={onTriggerLiveForecast}
            disabled={isLoadingLive}
            className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-emerald-600/20 disabled:opacity-50 transition-all cursor-pointer"
            title="Trigger live GraphCast + Satellite ResNet-18 pipeline"
          >
            <Sparkles className={`w-3.5 h-3.5 ${isLoadingLive ? 'animate-spin' : ''}`} />
            {isLoadingLive ? 'Running AI...' : 'Run Live AI'}
          </button>
        )}
      </div>

      {/* Right Stats & Status */}
      <div className="flex items-center gap-4 text-xs">
        <div className="hidden md:flex items-center gap-2 text-gray-400">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span>Feed: Live IODC</span>
        </div>
        <div className="bg-gray-900 border border-gray-800 px-2.5 py-1 rounded-md text-gray-300 flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-blue-400" />
          <span>PostGIS / Spatial Engine</span>
        </div>
      </div>
    </header>
  );
}
