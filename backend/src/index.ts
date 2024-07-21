/**
 * Import function triggers from their respective submodules:
 *
 * import {onCall} from "firebase-functions/v2/https";
 * import {onDocumentWritten} from "firebase-functions/v2/firestore";
 *
 * See a full list of supported triggers at https://firebase.google.com/docs/functions
 */

import { initializeApp } from 'firebase-admin/app';
import { logger } from 'firebase-functions/v2';
import { HttpsError, onCall } from 'firebase-functions/v2/https';
import pl from 'nodejs-polars';

const app = initializeApp({
	storageBucket: 'durham-river-level.appspot.com',
});

import type { File } from '@google-cloud/storage';
import { getStorage } from 'firebase-admin/storage';

export const getData = onCall(
	{
		region: 'europe-west2',
		cors: ['*'],
	},
	async (_request) => {
		try {
			const bucket = getStorage(app).bucket();

			const downloadDataFromBucket = async (file: File) => {
				logger.info(`Downloading ${file.name} from bucket`);
				const fileData = await file.download();
				const buffer = fileData[0];
				return pl.readParquet(buffer);
			};

			const [levelForecast, levelObservation, weatherForecast] =
				await Promise.all([
					downloadDataFromBucket(bucket.file('prediction/prediction.parquet')),
					downloadDataFromBucket(bucket.file('prediction/observation.parquet')),
					downloadDataFromBucket(
						bucket.file('weather/latest_forecast.parquet')
					),
				]);

			logger.info('All data downloaded successfully');

			return 'Hello World';

			// return { levelForecast, levelObservation, weatherForecast };
		} catch (e) {
			logger.error(e);
			throw new HttpsError('internal', 'An error occurred while fetching data');
		}
	}
);
