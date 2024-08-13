import { useForecastRiverLevel, useObservedRiverLevel } from '@/hooks';
import { useEffect, useState } from 'react';
import type { RiverLevel } from '@/types';

type ChartData = {
  data: RiverLevel[] | undefined;
  error: unknown;
}

export function useChartData() {
  const { data: observedLevelData, error: observedError } = useObservedRiverLevel();
  const { data: forecastLevelData, error: forecastError } = useForecastRiverLevel();

  const [chartData, setChartData] = useState<ChartData>({
    data: undefined,
    error: null,
  });

  useEffect(() => {
    if (observedLevelData && forecastLevelData) {
      const last_observed = observedLevelData[observedLevelData.length - 1];

      setChartData({
        data: [
          ...observedLevelData,
          {
            timestamp: last_observed.timestamp,
            predicted: forecastLevelData[0].predicted,
            std: 0,
          },
          ...forecastLevelData,
        ],
        error: null,
      });
    } else {
      setChartData({
        data: undefined,
        error: observedError ?? forecastError,
      });
    }
  }, [observedLevelData, forecastLevelData, observedError, forecastError]);

  return chartData;
}