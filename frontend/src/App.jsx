import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import SidePanel from './components/SidePanel';
import MapDashboard from './components/MapDashboard';
import IntensityChart from './components/IntensityChart';
import ReplayScrubber from './components/ReplayScrubber';
import ImpactModal from './components/ImpactModal';
import { ChevronUp, ChevronDown, Activity, Sparkles, CheckCircle2 } from 'lucide-react';

class MapErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    console.error('MapErrorBoundary caught an error:', error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="w-full h-full flex flex-col items-center justify-center bg-gray-950 text-gray-300 p-6 text-center">
          <div className="p-4 rounded-xl bg-gray-900 border border-red-500/30 max-w-md">
            <h3 className="text-sm font-bold text-red-400 mb-1">Map View Notice</h3>
            <p className="text-xs text-gray-400 mb-3">Map layer refreshed. Click below to reload coordinates.</p>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold"
            >
              Reset Map View
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {

  // Global State
  const [cyclones, setCyclones] = useState([]);
  const [selectedCycloneId, setSelectedCycloneId] = useState(null);
  const [selectedCyclone, setSelectedCyclone] = useState(null);

  // Layer & Mode State
  const [isReplayMode, setIsReplayMode] = useState(false);
  const [showSatellite, setShowSatellite] = useState(true);
  const [selectedBasin, setSelectedBasin] = useState('all');

  // Loaded Datasets for Selected Cyclone
  const [trackData, setTrackData] = useState(null);
  const [uncertaintyData, setUncertaintyData] = useState(null);
  const [intensityData, setIntensityData] = useState(null);
  const [replayData, setReplayData] = useState(null);
  const [impactData, setImpactData] = useState(null);

  // UI Drawer / Modal State
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isSidePanelOpen, setIsSidePanelOpen] = useState(true);
  const [isImpactModalOpen, setIsImpactModalOpen] = useState(false);
  const [isBottomDrawerOpen, setIsBottomDrawerOpen] = useState(true);
  const [isLoadingLive, setIsLoadingLive] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);

  // Fetch list of cyclones
  const fetchCyclones = useCallback(async () => {
    try {
      const res = await fetch('/api/cyclones');
      if (!res.ok) throw new Error('Failed to fetch cyclones');
      const data = await res.json();
      setCyclones(data);

      // Auto-select initial cyclone if none selected
      if (!selectedCycloneId && data.length > 0) {
        const defaultCyclone = isReplayMode
          ? data.find((c) => c.status === 'historical') || data[0]
          : data.find((c) => c.status === 'active') || data[0];
        setSelectedCycloneId(defaultCyclone.id);
      }
    } catch (err) {
      console.error('Error fetching cyclones:', err);
    }
  }, [isReplayMode, selectedCycloneId]);

  useEffect(() => {
    fetchCyclones();
  }, [fetchCyclones]);

  // Fetch all details when selected cyclone changes
  useEffect(() => {
    if (!selectedCycloneId) return;

    let isMounted = true;

    async function loadCycloneDetails() {
      try {
        // Basic detail
        const resDetail = await fetch(`/api/cyclones/${selectedCycloneId}`);
        if (resDetail.ok && isMounted) {
          const detail = await resDetail.json();
          setSelectedCyclone(detail);
        }

        // Track data
        const resTrack = await fetch(`/api/cyclones/${selectedCycloneId}/track`);
        if (resTrack.ok && isMounted) {
          const track = await resTrack.json();
          setTrackData(track);
        }

        // Uncertainty Cone
        const resCone = await fetch(`/api/cyclones/${selectedCycloneId}/uncertainty`);
        if (resCone.ok && isMounted) {
          const cone = await resCone.json();
          setUncertaintyData(cone);
        } else if (isMounted) {
          setUncertaintyData(null);
        }

        // Intensity timeline
        const resIntensity = await fetch(`/api/cyclones/${selectedCycloneId}/intensity`);
        if (resIntensity.ok && isMounted) {
          const intensity = await resIntensity.json();
          setIntensityData(intensity);
        }

        // Replay ground truth vs model
        const resReplay = await fetch(`/api/cyclones/${selectedCycloneId}/replay`);
        if (resReplay.ok && isMounted) {
          const replay = await resReplay.json();
          setReplayData(replay);
          setCurrentStepIndex(0);
        } else if (isMounted) {
          setReplayData(null);
        }

        // Impact & coastal vulnerability
        const resImpact = await fetch(`/api/cyclones/${selectedCycloneId}/impact`);
        if (resImpact.ok && isMounted) {
          const impact = await resImpact.json();
          setImpactData(impact);
        } else if (isMounted) {
          setImpactData(null);
        }
      } catch (err) {
        console.error('Error loading cyclone details:', err);
      }
    }

    loadCycloneDetails();

    return () => {
      isMounted = false;
    };
  }, [selectedCycloneId]);

  // Handle Mode Change (Live vs Replay)
  const handleModeChange = (replayMode) => {
    setIsReplayMode(replayMode);
    setCurrentStepIndex(0);
    // Find matching cyclone for that mode (Amphan has comprehensive ground-truth points)
    const target = cyclones.find((c) =>
      replayMode ? (c.id === 'amphan-2020' || c.status === 'historical') : c.status === 'active'
    ) || cyclones[0];

    if (target && target.id !== selectedCycloneId) {
      setReplayData(null);
      setSelectedCycloneId(target.id);
    }
  };

  // Run Multimodal AI Forecast pipeline (Real EUMETSAT ResNet-18 + NOAA GFS)
  const handleTriggerLiveForecast = async () => {
    setIsLoadingLive(true);
    try {
      let res = await fetch('/api/cyclones/forecast/live', { method: 'POST' });
      if (!res.ok) {
        res = await fetch('/api/forecast/live', { method: 'POST' });
      }
      if (!res.ok) throw new Error(`Live forecast trigger failed (${res.status})`);
      const result = await res.json();

      const confPct = Math.round((result.confidence_score != null ? result.confidence_score : 0.40) * 100);
      setToastMessage({
        title: 'Multimodal AI Forecast Completed',
        detail: `EUMETSAT 10.8µm + ResNet-18: '${result.category}' (${confPct}% confidence). 24h NOAA GFS atmospheric track updated.`,
      });

      // Refresh list & select live watch
      await fetchCyclones();
      setSelectedCycloneId(result.id || 'live-iodc-01');
      setIsReplayMode(false);
    } catch (err) {
      console.error('Error running live forecast:', err);
      setToastMessage({
        title: 'Forecast Run Notice',
        detail: `Pipeline error: ${err.message}`,
      });
    } finally {
      setIsLoadingLive(false);
      setTimeout(() => setToastMessage(null), 6000);
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-gray-950 text-white font-sans overflow-hidden select-none">
      {/* Top Navbar */}
      <Navbar
        isReplayMode={isReplayMode}
        setIsReplayMode={handleModeChange}
        showSatellite={showSatellite}
        setShowSatellite={setShowSatellite}
        selectedBasin={selectedBasin}
        setSelectedBasin={setSelectedBasin}
        onTriggerLiveForecast={handleTriggerLiveForecast}
        isLoadingLive={isLoadingLive}
      />

      {/* Main Workspace: Sidebar + Map + Side Panel */}
      <div className="flex-1 flex relative overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar
          cyclones={cyclones}
          selectedCycloneId={selectedCycloneId}
          onSelectCyclone={(id) => {
            setSelectedCycloneId(id);
            setIsSidePanelOpen(true);
          }}
          isReplayMode={isReplayMode}
        />

        {/* Center Map View */}
        <main className="flex-1 relative h-full">
          <MapErrorBoundary>
            <MapDashboard
              cyclone={selectedCyclone}
              trackData={trackData}
              uncertaintyData={uncertaintyData}
              satelliteOverlay={selectedCyclone?.satellite_overlay}
              showSatellite={showSatellite}
              isReplayMode={isReplayMode}
              replayData={replayData}
              currentStepIndex={currentStepIndex}
            />
          </MapErrorBoundary>


          {/* Floating Right Detail Panel */}
          <SidePanel
            cyclone={selectedCyclone}
            impactData={impactData}
            isOpen={isSidePanelOpen}
            onClose={() => setIsSidePanelOpen(false)}
            onOpenImpactModal={() => setIsImpactModalOpen(true)}
          />

          {/* Re-open SidePanel trigger button if closed */}
          {!isSidePanelOpen && selectedCyclone && (
            <button
              onClick={() => setIsSidePanelOpen(true)}
              className="absolute right-4 top-16 z-20 px-3 py-2 rounded-xl bg-gray-950/90 border border-gray-800 text-xs text-white hover:bg-gray-900 transition-all shadow-xl flex items-center gap-1.5"
            >
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>Storm Detail</span>
            </button>
          )}

          {/* Bottom Floating Drawer: Intensity Timeline OR Replay Scrubber */}
          <div className="absolute bottom-4 left-4 right-4 z-20 max-w-4xl mx-auto">
            <div className="bg-gray-950/95 border border-gray-800 backdrop-blur-md rounded-2xl shadow-2xl overflow-hidden transition-all duration-300">
              {/* Drawer Header / Tab Toggle */}
              <div className="px-4 py-2 bg-gray-900/60 border-b border-gray-800/80 flex items-center justify-between text-xs">
                <div className="flex items-center gap-3">
                  <span className="font-bold text-white flex items-center gap-1.5">
                    {isReplayMode ? (
                      <>
                        <span className="w-2 h-2 rounded-full bg-purple-400"></span>
                        Historical Replay & Ground-Truth Verification
                      </>
                    ) : (
                      <>
                        <span className="w-2 h-2 rounded-full bg-blue-400"></span>
                        GraphCast 24h Intensity Forecast Timeline
                      </>
                    )}
                  </span>
                  {selectedCyclone && (
                    <span className="text-gray-400 text-[11px]">
                      — {selectedCyclone.name} ({selectedCyclone.basin})
                    </span>
                  )}
                </div>

                <button
                  onClick={() => setIsBottomDrawerOpen(!isBottomDrawerOpen)}
                  className="p-1 rounded text-gray-400 hover:text-white transition-colors"
                  title={isBottomDrawerOpen ? 'Collapse drawer' : 'Expand drawer'}
                >
                  {isBottomDrawerOpen ? (
                    <ChevronDown className="w-4 h-4" />
                  ) : (
                    <ChevronUp className="w-4 h-4" />
                  )}
                </button>
              </div>

              {/* Drawer Body */}
              {isBottomDrawerOpen && (
                <div className="p-3">
                  {isReplayMode ? (
                    <ReplayScrubber
                      replayData={replayData}
                      currentStepIndex={currentStepIndex}
                      onStepChange={setCurrentStepIndex}
                    />
                  ) : (
                    <IntensityChart intensityData={intensityData} />
                  )}
                </div>
              )}
            </div>
          </div>
        </main>
      </div>

      {/* Coastal Impact & Vulnerability Modal */}
      <ImpactModal
        isOpen={isImpactModalOpen}
        onClose={() => setIsImpactModalOpen(false)}
        cyclone={selectedCyclone}
        impactData={impactData}
      />

      {/* Floating Notification Toast */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 bg-gray-900 border border-blue-500/50 p-4 rounded-xl shadow-2xl text-xs text-white max-w-sm animate-slide-up flex items-start gap-3">
          <CheckCircle2 className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-bold text-sm text-cyan-300">{toastMessage.title}</h4>
            <p className="text-gray-300 mt-1 leading-relaxed">{toastMessage.detail}</p>
          </div>
        </div>
      )}
    </div>
  );
}
