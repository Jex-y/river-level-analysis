import * as logger from 'firebase-functions/logger';
import { onRequest } from 'firebase-functions/v2/https';

import { MongoClient } from 'mongodb';

const dbName = 'riverdata';
const collectionName = 'spills';
let _client: MongoClient | null = null;

const getClient = () => {
	if (!_client) {
		const url = process.env.DB_URI;
		if (!url) {
			throw new Error('DB_URI environment variable is required');
		}
		_client = new MongoClient(url);
	}

	return _client;
};

const fetch_data = async () => {
	const client = getClient();
	const db = client.db(dbName);
	const collection = db.collection(collectionName);

	logger.info('Fetching sewage leaks');

	const results = await collection
		.find(
			{
				event_end: {
					$gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
				},
			},
			{
				projection: { _id: 0 },
			}
		)
		.toArray();

	logger.info(`Found ${results.length} sewage leaks`);
	return results;
};

export const getSewageLeaks = onRequest(
	{
		region: 'europe-west2',
		cors: '*',
	},
	async (_req, res) => {
		res.set('Cache-Control', 'public, max-age=300, s-maxage=900');
		res.json(await fetch_data());
	}
);
