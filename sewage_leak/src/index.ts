import * as logger from 'firebase-functions/logger';
import { onRequest } from 'firebase-functions/v2/https';

import {
	type Document,
	type FindOptions,
	MongoClient,
	type WithId,
} from 'mongodb';

const url = process.env.DB_URI;
if (!url) {
	throw new Error('DB_URI environment variable is required');
}

const client = new MongoClient(url);
const dbName = 'riverdata';
const collectionName = 'spills';

const formatEvent = <T extends {}, V extends {}>(
	events: T[],
	new_keys: V
): (T & V)[] =>
	events.map((event: T) => ({
		...event,
		...new_keys,
	}));

const logQuery = async (
	query_name: string,
	promise: Promise<WithId<Document>[]>
): Promise<WithId<Document>[]> => {
	logger.info(`Query ${query_name} started`);
	const results = await promise;
	logger.info(`Query ${query_name} returned ${results.length} results`, {
		results,
	});
	return results;
};

const fetch_data = async () => {
	const db = client.db(dbName);
	const collection = db.collection(collectionName);

	const nearbySpillsPromise = collection
		.find(
			{
				'metadata.nearby': true,
				event_type: 'spill',
				event_end: {
					$gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
				},
			},
			{
				projection: { _id: 0 },
			} satisfies FindOptions
		)
		.toArray();

	const nearbyOfflinePromise = collection
		.find(
			{
				'metadata.nearby': true,
				event_type: 'monitor offline',
				event_end: {
					$gte: new Date(Date.now() - 24 * 60 * 60 * 1000),
				},
			},
			{
				projection: { _id: 0 },
			} satisfies FindOptions
		)
		.toArray();

	const distantSpillsPromise = collection
		.find(
			{
				'metadata.nearby': false,
				type: 'spill',
				event_end: {
					$gte: new Date(Date.now() - 24 * 60 * 60 * 1000),
				},
			},
			{
				projection: { _id: 0 },
			} satisfies FindOptions
		)
		.toArray();

	const [nearbySpills, nearbyOffline, distantSpills] = await Promise.all([
		logQuery('nearbySpills', nearbySpillsPromise),
		logQuery('nearbyOffline', nearbyOfflinePromise),
		logQuery('distantSpills', distantSpillsPromise),
	]);

	return [
		...formatEvent(nearbySpills, { severity: 'high' }),
		...formatEvent(nearbyOffline, { severity: 'medium' }),
		...formatEvent(distantSpills, { severity: 'medium' }),
	];
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
