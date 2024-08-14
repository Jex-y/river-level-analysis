import { storage } from '@/firebase';
import {
	ForecastRiverLevelSchema,
	ObservedRiverLevelSchema,
	SpillSiteSchema,
	WeatherForecastSchema,
} from '@/types';
import { getBytes, ref } from 'firebase/storage';
import useSwr, { type SWRConfiguration } from 'swr';
import { z } from 'zod';

const createSchemaHttpFetcher =
	<T extends z.ZodTypeAny>(schema: T) =>
	async (url: string): Promise<z.infer<T>> =>
		fetch(url)
			.then((res) => res.json())
			.then(schema.parseAsync);

const createSchemaBucketFetcher =
	<T extends z.ZodTypeAny>(schema: T) =>
	async (path: string): Promise<z.infer<T>[]> => {
		const path_ref = ref(storage, path);
		const bytes = await getBytes(path_ref);
		const data = JSON.parse(new TextDecoder().decode(bytes));
		return schema.parse(data);
	};

const createHttpApiHook =
	<T extends z.ZodTypeAny>(url: string, schema: T, config: SWRConfiguration) =>
	() =>
		useSwr<z.infer<T>>(url, createSchemaHttpFetcher(schema), config);

const createBucketHook =
	<T extends z.ZodTypeAny>(path: string, schema: T, config: SWRConfiguration) =>
	() =>
		useSwr<z.infer<T>>(path, createSchemaBucketFetcher(schema), config);

const shared_config = {
	onError: (error: Error) => {
		if (error instanceof z.ZodError) {
			console.error('Failed to parse data from API. Issues:', error.issues);
		} else {
			console.error('Failed to fetch data from API:', error);
		}
	},
} satisfies SWRConfiguration;

export const useObservedRiverLevel = createBucketHook(
	'prediction/observation.json',
	z.array(ObservedRiverLevelSchema),
	{
		...shared_config,
		refreshInterval: 1000 * 60 * 5,
	}
);

export const useForecastRiverLevel = createBucketHook(
	'prediction/prediction.json',
	z.array(ForecastRiverLevelSchema),
	{
		...shared_config,
		refreshInterval: 1000 * 60 * 5,
	}
);

export const useWeatherForecast = createBucketHook(
	'weather/latest_forecast.json',
	z.array(WeatherForecastSchema),
	{
		...shared_config,
		refreshInterval: 1000 * 60 * 5,
	}
);

export const useSpillSites = createHttpApiHook(
	'https://getspillsites-fjdislxqya-nw.a.run.app',
	z.array(SpillSiteSchema),
	{
		...shared_config,
		refreshInterval: 1000 * 60 * 15,
	}
);
