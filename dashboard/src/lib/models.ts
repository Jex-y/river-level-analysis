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
