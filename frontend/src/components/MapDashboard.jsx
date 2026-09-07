import React, { useEffect, useMemo, useRef } from 'react';
import {
  MapContainer,
  TileLayer,
  Polyline,
  CircleMarker,
  GeoJSON,
  ImageOverlay,
  Tooltip,
  useMap,
} from 'react-leaflet';
import { getCategoryColor } from '../utils/constants';

/**
 * Validates a single [lat, lon] pair.
 */
function isValidCoord(c) {
  return Array.isArray(c) && c.length === 2 && typeof c[0] === 'number' && typeof c[1] === 'number' && !isNaN(c[0]) && !isNaN(c[1]);
}

/**
 * Controller to smoothly pan the map when the active cyclone or replay step changes.
 */
function MapController({ center, zoom }) {
  const map = useMap();
  const prevCenterRef = useRef(null);

  useEffect(() => {
    if (!isValidCoord(center)) return;

    // Avoid redundant flyTo calls if within 0.05 degrees
    const prev = prevCenterRef.current;
    if (prev && Math.abs(prev[0] - center[0]) < 0.05 && Math.abs(prev[1] - center[1]) < 0.05) {
      return;
    }

    prevCenterRef.current = center;
    try {
      map.flyTo(center, zoom || 6, { duration: 0.8 });
    } catch {
      try {
        map.setView(center, zoom || 6);
      } catch {}
    }
  }, [center, zoom, map]);

  return null;
}

export default function MapDashboard({
  cyclone,
  trackData,
  uncertaintyData,
  satelliteOverlay,
  showSatellite,
  isReplayMode,
  replayData,
  currentStepIndex,
}) {
  // Safe center determination
  const center = useMemo(() => {
    if (isReplayMode && replayData?.steps?.length > 0) {
      const idx = Math.min(Math.max(0, currentStepIndex || 0), replayData.steps.length - 1);
      const step = replayData.steps[idx];
      if (step && isValidCoord([step.actual_lat, step.actual_lon])) {
        return [step.actual_lat, step.actual_lon];
      }
    }
    if (cyclone && isValidCoord([cyclone.current_lat, cyclone.current_lon])) {
      return [cyclone.current_lat, cyclone.current_lon];
    }
    return [18.0, 85.0]; // Default North Indian Ocean
  }, [isReplayMode, replayData, currentStepIndex, cyclone]);

  // Points for live forecast track
  const trackPoints = useMemo(() => {
    if (!trackData?.points || !Array.isArray(trackData.points)) return [];
    return trackData.points.filter((p) => isValidCoord([p.lat, p.lon]));
  }, [trackData]);

  const polylineCoords = useMemo(() => {
    return trackPoints.map((p) => [p.lat, p.lon]);
  }, [trackPoints]);

  // Replay steps
  const replaySteps = useMemo(() => {
    if (!replayData?.steps || !Array.isArray(replayData.steps)) return [];
    return replayData.steps.filter((s) => isValidCoord([s.actual_lat, s.actual_lon]));
  }, [replayData]);

  const safeStepIndex = Math.min(Math.max(0, currentStepIndex || 0), Math.max(0, replaySteps.length - 1));
  const currentReplayStep = replaySteps[safeStepIndex] || null;

  // Actual track coords up to current step
  const actualLineCoords = useMemo(() => {
    if (!replaySteps.length) return [];
    return replaySteps
      .slice(0, safeStepIndex + 1)
      .map((s) => [s.actual_lat, s.actual_lon])
      .filter(isValidCoord);
  }, [replaySteps, safeStepIndex]);

  // Predicted track coords up to current step
  const predictedLineCoords = useMemo(() => {
    if (!replaySteps.length) return [];
    return replaySteps
      .slice(0, safeStepIndex + 1)
      .map((s) => [s.predicted_lat, s.predicted_lon])
      .filter(isValidCoord);
  }, [replaySteps, safeStepIndex]);

  // Full actual path (for faint background path)
  const fullActualLineCoords = useMemo(() => {
    if (!replaySteps.length) return [];
    return replaySteps.map((s) => [s.actual_lat, s.actual_lon]).filter(isValidCoord);
  }, [replaySteps]);

  // Satellite bounds
  const satBounds = useMemo(() => {
    if (satelliteOverlay?.bounds && Array.isArray(satelliteOverlay.bounds)) {
      return satelliteOverlay.bounds;
    }
    return [[0.0, 55.0], [30.0, 100.0]];
  }, [satelliteOverlay]);

  const satUrl = satelliteOverlay?.image_url || '/api/satellite/latest_satellite_ir.png';

  // Active cyclone mode key to ensure stable leaflet mounting
  const mapKey = `cyclocast-map-${isReplayMode ? 'replay' : 'live'}-${cyclone?.id || 'default'}`;

  return (
    <div className="w-full h-full relative bg-gray-950 overflow-hidden">
      <MapContainer
        key={mapKey}
        center={center}
        zoom={5}
        scrollWheelZoom={true}
        className="w-full h-full z-10"
        attributionControl={false}
      >
        <MapController center={center} zoom={cyclone ? 6 : 5} />

        {/* High-Contrast Dark Map Tiles (Free, No API Key, No Watermark) */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
          maxZoom={16}
        />
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
          maxZoom={16}
          zIndex={11}
        />


        {/* Satellite IR Overlay (Shown in Live Mode or when requested) */}
        {showSatellite && !isReplayMode && satUrl && (
          <ImageOverlay
            key={`satellite-overlay-${cyclone?.id || 'live'}`}
            url={satUrl}
            bounds={satBounds}
            opacity={0.65}
            zIndex={12}
          />
        )}

        {/* Uncertainty Cone GeoJSON Polygon (Live Mode) */}
        {!isReplayMode && uncertaintyData && (
          <GeoJSON
            key={`cone-${cyclone?.id}-${uncertaintyData?.properties?.forecast_run_id || 1}`}
            data={uncertaintyData}
            style={{
              color: '#f59e0b',
              weight: 1.5,
              opacity: 0.8,
              fillColor: '#f59e0b',
              fillOpacity: 0.15,
              dashArray: '5, 5',
            }}
          />
        )}

        {/* ========================================================================= */}
        {/* LIVE FORECAST MODE LAYERS                                                 */}
        {/* ========================================================================= */}
        {!isReplayMode && (
          <>
            {/* Forecast Track Polyline */}
            {polylineCoords.length >= 2 && (
              <Polyline
                key={`live-polyline-${cyclone?.id}`}
                positions={polylineCoords}
                pathOptions={{
                  color: '#38bdf8',
                  weight: 3,
                  dashArray: '6, 6',
                  opacity: 0.9,
                }}
              />
            )}

            {/* Track Point Markers */}
            {trackPoints.map((pt, idx) => {
              const catColor = getCategoryColor(pt.category);
              const isCurrent = pt.horizon_hours === 0;

              return (
                <React.Fragment key={`live-track-marker-group-${idx}`}>
                  {isCurrent && (
                    <CircleMarker
                      key={`live-pulse-${idx}`}
                      center={[pt.lat, pt.lon]}
                      radius={16}
                      pathOptions={{
                        color: '#38bdf8',
                        fillColor: '#38bdf8',
                        fillOpacity: 0.15,
                        weight: 2,
                        dashArray: '3, 3',
                      }}
                    />
                  )}

                  <CircleMarker
                    key={`live-point-${idx}`}
                    center={[pt.lat, pt.lon]}
                    radius={isCurrent ? 9 : 7}
                    pathOptions={{
                      color: isCurrent ? '#38bdf8' : '#ffffff',
                      fillColor: catColor,
                      fillOpacity: 0.95,
                      weight: isCurrent ? 3 : 2,
                    }}
                  >
                    <Tooltip direction="top" offset={[0, -8]} opacity={0.95}>
                      <div className="text-xs font-sans text-gray-900 leading-tight">
                        <div className="font-bold flex items-center gap-1">
                          <span style={{ color: catColor }}>●</span>
                          {pt.horizon_hours === 0 ? 'Current Eye' : `+${pt.horizon_hours}h Forecast`}
                        </div>
                        <div>Category: <b>{pt.category}</b></div>
                        <div>Wind: <b>{Math.round(pt.wind_speed_kt)} kt</b> ({Math.round(pt.wind_speed_kt * 1.852)} km/h)</div>
                        <div>Pressure: <b>{Math.round(pt.pressure_hpa)} hPa</b></div>
                        <div className="text-[10px] text-gray-500">[{pt.lat.toFixed(2)}°N, {pt.lon.toFixed(2)}°E]</div>
                      </div>
                    </Tooltip>
                  </CircleMarker>
                </React.Fragment>
              );
            })}
          </>
        )}

        {/* ========================================================================= */}
        {/* REPLAY MODE LAYERS (GROUND TRUTH VS MODEL PREDICTION)                     */}
        {/* ========================================================================= */}
        {isReplayMode && replaySteps.length > 0 && (
          <>
            {/* Full historical ground truth trace (dimmed) */}
            {fullActualLineCoords.length >= 2 && (
              <Polyline
                key={`replay-full-actual-${cyclone?.id}`}
                positions={fullActualLineCoords}
                pathOptions={{
                  color: '#10b981',
                  weight: 2,
                  opacity: 0.25,
                }}
              />
            )}

            {/* Traveled Actual Path (Ground Truth) */}
            {actualLineCoords.length >= 2 && (
              <Polyline
                key={`replay-actual-traveled-${cyclone?.id}-${safeStepIndex}`}
                positions={actualLineCoords}
                pathOptions={{
                  color: '#10b981',
                  weight: 3.5,
                  opacity: 0.9,
                }}
              />
            )}

            {/* Traveled Predicted Path (Model Output) */}
            {predictedLineCoords.length >= 2 && (
              <Polyline
                key={`replay-predicted-traveled-${cyclone?.id}-${safeStepIndex}`}
                positions={predictedLineCoords}
                pathOptions={{
                  color: '#a855f7',
                  weight: 3,
                  dashArray: '5, 5',
                  opacity: 0.9,
                }}
              />
            )}

            {/* Current Step Error Vector (Line connecting actual and predicted) */}
            {currentReplayStep &&
              isValidCoord([currentReplayStep.actual_lat, currentReplayStep.actual_lon]) &&
              isValidCoord([currentReplayStep.predicted_lat, currentReplayStep.predicted_lon]) && (
                <Polyline
                  key={`replay-error-vector-${safeStepIndex}`}
                  positions={[
                    [currentReplayStep.actual_lat, currentReplayStep.actual_lon],
                    [currentReplayStep.predicted_lat, currentReplayStep.predicted_lon],
                  ]}
                  pathOptions={{
                    color: '#ef4444',
                    weight: 2,
                    dashArray: '3, 4',
                    opacity: 0.9,
                  }}
                />
              )}

            {/* Markers for historical steps up to current step */}
            {replaySteps.slice(0, safeStepIndex + 1).map((s, idx) => (
              <React.Fragment key={`replay-step-marker-group-${idx}`}>
                {/* Actual Ground Truth Marker */}
                {isValidCoord([s.actual_lat, s.actual_lon]) && (
                  <CircleMarker
                    key={`replay-actual-pt-${idx}`}
                    center={[s.actual_lat, s.actual_lon]}
                    radius={idx === safeStepIndex ? 8 : 4}
                    pathOptions={{
                      color: '#ffffff',
                      fillColor: '#10b981',
                      fillOpacity: 0.9,
                      weight: idx === safeStepIndex ? 2 : 1,
                    }}
                  >
                    <Tooltip direction="top" offset={[0, -6]}>
                      <div className="text-xs font-sans text-gray-900">
                        <b className="text-emerald-600">Actual: {s.actual_category || 'Storm'}</b>
                        <div>Wind: {Math.round(s.actual_wind_kt || 0)} kt</div>
                        <div>Press: {Math.round(s.actual_pressure_hpa || 0)} hPa</div>
                      </div>
                    </Tooltip>
                  </CircleMarker>
                )}

                {/* Predicted Marker */}
                {isValidCoord([s.predicted_lat, s.predicted_lon]) && (
                  <CircleMarker
                    key={`replay-pred-pt-${idx}`}
                    center={[s.predicted_lat, s.predicted_lon]}
                    radius={idx === safeStepIndex ? 8 : 4}
                    pathOptions={{
                      color: '#ffffff',
                      fillColor: '#a855f7',
                      fillOpacity: 0.9,
                      weight: idx === safeStepIndex ? 2 : 1,
                    }}
                  >
                    <Tooltip direction="bottom" offset={[0, 6]}>
                      <div className="text-xs font-sans text-gray-900">
                        <b className="text-purple-600">Predicted: {s.predicted_category || 'Storm'}</b>
                        <div>Wind: {Math.round(s.predicted_wind_kt || 0)} kt</div>
                        <div>Press: {Math.round(s.predicted_pressure_hpa || 0)} hPa</div>
                      </div>
                    </Tooltip>
                  </CircleMarker>
                )}
              </React.Fragment>
            ))}
          </>
        )}
      </MapContainer>

      {/* Map Legend Overlay */}
      <div className="absolute top-4 left-4 z-20 bg-gray-950/85 backdrop-blur-md border border-gray-800 rounded-xl p-3 text-xs text-gray-300 shadow-xl pointer-events-auto space-y-2 max-w-xs">
        <div className="font-semibold text-white flex items-center justify-between">
          <span>Map Legend</span>
          <span className="text-[10px] text-gray-400">
            {isReplayMode ? 'Model vs. Ground Truth' : 'NOAA GFS + Satellite AI'}
          </span>
        </div>

        {!isReplayMode ? (
          <div className="space-y-1.5 text-[11px]">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-cyan-400 border border-white inline-block"></span>
              <span>Predicted Cyclone Eye (0h)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-5 h-0.5 border-t-2 border-dashed border-sky-400 inline-block"></span>
              <span>Predicted Track Line (+6h to +24h)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3.5 h-3.5 bg-amber-500/30 border border-dashed border-amber-400 rounded inline-block"></span>
              <span>Uncertainty Cone (IMD Model)</span>
            </div>
            {showSatellite && (
              <div className="flex items-center gap-2">
                <span className="w-3.5 h-3.5 bg-gradient-to-r from-blue-900 to-indigo-700 rounded border border-gray-600 inline-block"></span>
                <span>Meteosat-9 IR Thermal Channel</span>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-1.5 text-[11px]">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-emerald-500 border border-white inline-block"></span>
              <span>Actual Best-Track (IMD Ground Truth)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-purple-500 border border-white inline-block"></span>
              <span>CycloCast AI Predicted Track</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-5 h-0.5 border-t-2 border-dashed border-red-500 inline-block"></span>
              <span>Track Error Vector (km Discrepancy)</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
