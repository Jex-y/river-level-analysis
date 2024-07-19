import { type ReadableAtom, atom, computed } from 'nanostores';
import type {
	ForecastRiverLevel,
	ObservedRiverLevel,
	WeatherForecast,
} from './lib/models';
import type { Parameters } from './lib/parametersConfig';

// export type Theme = 'light' | 'dark'; // | 'darc';

// export const theme = atom<Theme | null>('light');

export const levelForecastStore = atom<ForecastRiverLevel[] | undefined>(
	undefined
);
export const levelObservationStore = atom<ObservedRiverLevel[] | undefined>(
	undefined
);

export const weatherForecastStore = atom<WeatherForecast[] | undefined>(
	undefined
);

export const currentLevelObservationStore = computed(
	[levelObservationStore],
	(observed) => {
		if (observed === undefined) {
			return undefined;
		}
		return observed[observed.length - 1];
	}
);

export const currentWeatherForecastStore = computed(
	[weatherForecastStore],
	(weather) => {
		if (weather === undefined) {
			return undefined;
		}
		return weather[0];
	}
);

export const currentConditionsStore = computed<
	Parameters | undefined,
	[
		ReadableAtom<WeatherForecast | undefined>,
		ReadableAtom<ObservedRiverLevel | undefined>,
	]
>(
	[currentWeatherForecastStore, currentLevelObservationStore],
	(weather, levelObservation) => {
		if (weather === undefined || levelObservation === undefined) {
			return undefined;
		}

		return {
			temperature: weather.temperature,
			temperatureApparent: weather.temperatureApparent,
			uvIndex: weather.uvIndex,
			visibility: weather.visibility,
			windGust: weather.windGust,
			windSpeed: weather.windSpeed,
			riverLevel: levelObservation.observed,
		} satisfies Parameters;
	}
);

// export const currentRiverLevel = computed(
// 	[currentForecastRiverLevel, currentObservedRiverLevel],
// 	(forecast, observed) => {
// 		if (forecast === undefined && observed === undefined) {
// 			return undefined;
// 		}
// 		if (forecast === undefined) {
// 			return observed;
// 		}
// 		if (observed === undefined) {
// 			return forecast;
// 		}
// 		return [...observed, ...forecast] as RiverLevel[];
// 	}
// );
