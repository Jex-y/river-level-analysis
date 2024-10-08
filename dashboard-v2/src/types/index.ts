import { z } from 'zod';

export const ForecastRiverLevelSchema = z.object({
	timestamp: z.number().positive(),
	predicted: z.number().nonnegative(),
	std: z.number().nonnegative(),
});
export type ForecastRiverLevel = z.infer<typeof ForecastRiverLevelSchema>;

export const ObservedRiverLevelSchema = z.object({
	timestamp: z.number().positive(),
	observed: z.number().nonnegative(),
});
export type ObservedRiverLevel = z.infer<typeof ObservedRiverLevelSchema>;

export const RiverLevelSchema = z.union([
	ForecastRiverLevelSchema,
	ObservedRiverLevelSchema,
]);
export type RiverLevel = z.infer<typeof RiverLevelSchema>;

export const WeatherForecastSchema = z.object({
	time: z.coerce.date(),
	temperature: z.number(),
	temperatureApparent: z.number(),
	uvIndex: z.nullable(z.number().nonnegative()),
	visibility: z.number().nonnegative(),
	windGust: z.number().nonnegative(),
	windSpeed: z.number().nonnegative(),
});
export type WeatherForecast = z.infer<typeof WeatherForecastSchema>;

export const SpillSiteSchema = z.object({
	metadata: z.object({
		site_name: z.string(),
		site_id: z.string(),
		nearby: z.boolean(),
	}),
	events: z.array(
		z.object({
			event_start: z.coerce.date(),
			event_end: z.coerce.date(),
			event_type: z.union([
				z.literal('spill'),
				z.literal('monitor offline'),
				z.literal('no recent spill'),
			]),
			event_duration_mins: z.number().positive(),
		})
	),
});

export type SpillSite = z.infer<typeof SpillSiteSchema>;
