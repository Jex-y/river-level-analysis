import { useForecastRiverLevel, useObservedRiverLevel } from '@/hooks';
import type { ForecastRiverLevel, ObservedRiverLevel } from '@/types';
import { useEffect, useState } from 'react';
import type { RiverLevel } from '@/types';

type ChartData = {
	forecast?: ForecastRiverLevel[];
	observed?: ObservedRiverLevel[];
	error: unknown;
};

export function useChartData() {
	const { data: observedData, error: observedError } = useObservedRiverLevel();
	const { data: forecastData, error: forecastError } = useForecastRiverLevel();

	const [chartData, setChartData] = useState<ChartData>({
		error: null,
	});

	useEffect(() => {
		setChartData({
			forecast: forecastData,
			observed: observedData,
			error: observedError || forecastError,
		});
	}, [observedData, forecastData, observedError, forecastError]);

	return chartData;
}
