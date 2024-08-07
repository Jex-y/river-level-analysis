import { type ReadableAtom, atom, computed } from 'nanostores';
import type {
	ForecastRiverLevel,
	ObservedRiverLevel,
	WeatherForecast,
} from './lib/models';
import type { Parameters } from './lib/parametersConfig';

export const levelForecastStore = atom<ForecastRiverLevel[] | 'loading'>(
	'loading'
);
export const levelObservationStore = atom<ObservedRiverLevel[] | 'loading'>(
	'loading'
);

export const weatherForecastStore = atom<WeatherForecast[] | 'loading'>(
	'loading'
);

export const currentLevelObservationStore = computed(
	[levelObservationStore],
	(observed) => {
		console.log('Observed level:', observed);
		if (observed === 'loading') {
			return observed;
		}
		return observed[observed.length - 1];
	}
);

export const currentWeatherForecastStore = computed(
	[weatherForecastStore],
	(weather) => {
		if (weather === 'loading') {
			return weather;
		}
		return weather[0];
	}
);

export const currentConditionsStore = computed<
	Parameters | 'loading',
	[
		ReadableAtom<WeatherForecast | 'loading'>,
		ReadableAtom<ObservedRiverLevel | 'loading'>,
	]
>(
	[currentWeatherForecastStore, currentLevelObservationStore],
	(weather, levelObservation) => {
		if (weather === 'loading' || levelObservation === 'loading') {
			return 'loading';
		}

		console.log(levelObservation);

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
