import { z } from 'zod';

export type ForecastRiverLevel = {
	timestamp: number;
	predicted: number;
	ci: [number, number];
};

export type ObservedRiverLevel = {
	timestamp: number;
	observed: number;
};

export type RiverLevel = ForecastRiverLevel | ObservedRiverLevel;

// x = {
// 	time: '2024-07-19 15:00:00',
// 	precipitationProbability: 0,
// 	freezingRainIntensity: 0,
// 	iceAccumulation: 0,
// 	iceAccumulationLwe: 0,
// 	rainAccumulation: 0.0,
// 	rainAccumulationLwe: 0.0,
// 	rainIntensity: 0.0,
// 	sleetAccumulation: 0,
// 	sleetAccumulationLwe: 0,
// 	sleetIntensity: 0,
// 	snowAccumulation: 0,
// 	snowAccumulationLwe: 0,
// 	snowIntensity: 0,
// 	temperature: 24.38,
// 	temperatureApparent: 24.38,
// 	uvIndex: 1,
// 	visibility: 16.0,
// 	windGust: 10.38,
// 	windSpeed: 5.81,
// };

export type WeatherForecast = {
	time: Date;
	temperature: number;
	temperatureApparent: number;
	uvIndex: number;
	visibility: number;
	windGust: number;
	windSpeed: number;
};

export type Data = {
	levelForecast: ForecastRiverLevel[];
	levelObservation: ObservedRiverLevel[];
	weatherForecast: WeatherForecast[];
};

// export type SewageEvent = {
// 	metadata: {
// 		site_name: string;
// 		site_id: string;
// 		nearby: boolean;
// 		event_id: string;
// 	};
// 	event_start: Date;
// 	event_end: Date;
// 	event_type: 'spill' | 'monitor offline';
// 	severity: 'low' | 'medium' | 'high';
// };

export const SewageEventSchema = z.object({
	metadata: z.object({
		site_name: z.string(),
		site_id: z.string(),
		nearby: z.boolean(),
		event_id: z.string(),
	}),
	event_start: z.coerce.date(),
	event_end: z.coerce.date(),
	event_type: z.union([z.literal('spill'), z.literal('monitor offline')]),
});

export type SewageEvent = z.infer<typeof SewageEventSchema>;
