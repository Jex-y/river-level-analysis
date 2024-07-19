import { app } from '@/firebase/server';
import type { File } from '@google-cloud/storage';
import type { APIRoute } from 'astro';
import { getStorage } from 'firebase-admin/storage';

export const prerender = false;

export const GET: APIRoute = async () => {
	try {
		const bucket = getStorage(app).bucket();

		const downloadDataFromBucket = async (file: File) => {
			console.info(`Downloading ${file.name} from bucket`);
			const fileData = await file.download();
			return JSON.parse(fileData.toString());
		};

		const [levelForecast, levelObservation, weatherForecast] =
			await Promise.all([
				downloadDataFromBucket(bucket.file('prediction/prediction.json')),
				downloadDataFromBucket(bucket.file('prediction/observation.json')),
				downloadDataFromBucket(bucket.file('weather/latest_forecast.json')),
			]);

		return new Response(
			JSON.stringify({ levelForecast, levelObservation, weatherForecast }),
			{
				headers: {
					'Content-Type': 'application/json',
				},
			}
		);
	} catch (e) {
		console.error(e);
		return new Response('Error fetching data', { status: 500 });
	}
};
