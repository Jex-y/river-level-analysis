import { useEffect, useState } from 'react';

import { useObservedRiverLevel, useWeatherForecast } from '@/hooks';
import type { Parameters } from './types';

export function useParameters() {
        const observedRiverLevel = useObservedRiverLevel();
        const weatherForecast = useWeatherForecast();

        const [parameters, setParameters] = useState<Parameters | undefined>(
                undefined
        );

        useEffect(() => {
                if (observedRiverLevel.data && weatherForecast.data) {
                        const latestObservedRiverLevel = observedRiverLevel.data[0];
                        const currentWeather = weatherForecast.data[0];

                        setParameters({
                                riverLevel: latestObservedRiverLevel.value,
                                windSpeed: currentWeather.windSpeed,
                                windGust: currentWeather.windGust,
                                temperature: currentWeather.temperature,
                                temperatureApparent: currentWeather.temperatureApparent,
                                visibility: currentWeather.visibility,
                        });
                }
        }, [observedRiverLevel.data, weatherForecast.data]);

        return parameters;
}
