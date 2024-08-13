import { Status } from '@/components/ui/status-color-dot';
import type { SewageEvent } from '@/lib/models';

export const statusFn = (
	event: SewageEvent
): { status: Status; explanation: string } => {
	const in_last_week =
		new Date().getTime() - event.event_end.getTime() < 7 * 24 * 60 * 60 * 1000;
	const in_last_day =
		new Date().getTime() - event.event_end.getTime() < 24 * 60 * 60 * 1000;

	if (event.metadata.nearby) {
		if (event.event_type === 'spill') {
			if (in_last_day) {
				return {
					status: Status.Bad,
					explanation: 'Spill was nearby and in the last 24 hours.',
				};
			}
			if (in_last_week) {
				return {
					status: Status.Warning,
					explanation: 'Spill was nearby and in the last week.',
				};
			}
			return {
				status: Status.Good,
				explanation: 'Spill was more than a week ago.',
			};
		}

		if (event.event_type === 'monitor offline') {
			if (in_last_day) {
				return {
					status: Status.Warning,
					explanation:
						'Monitor was offline within the last 24 hours. Spills may not have been reported.',
				};
			}
			if (in_last_week) {
				return {
					status: Status.Warning,
					explanation:
						'Monitor was offline within the last week. Spills may not have been reported.',
				};
			}
			return {
				status: Status.Good,
				explanation:
					'Monitor was offline more than a week ago. Spills may not have been reported.',
			};
		}
	}

	if (event.event_type === 'spill') {
		if (in_last_day) {
			return {
				status: Status.Warning,
				explanation: 'Spill was not nearby and in the last 24 hours.',
			};
		}
		if (in_last_week) {
			return {
				status: Status.Good,
				explanation: 'Spill was not nearby and in the last week.',
			};
		}
		return {
			status: Status.Good,
			explanation: 'Spill was not nearby and more than a week ago.',
		};
	}

	if (event.event_type === 'monitor offline') {
		if (in_last_day) {
			return {
				status: Status.Good,
				explanation:
					'Monitor was offline within the last 24 hours. Spills may not have been reported.',
			};
		}
		if (in_last_week) {
			return {
				status: Status.Good,
				explanation:
					'Monitor was offline within the last week. Spills may not have been reported.',
			};
		}
		return {
			status: Status.Good,
			explanation:
				'Monitor was offline more than a week ago. Spills may not have been reported.',
		};
	}

	console.error('Unhandled event type', event);
	return { status: Status.Good, explanation: 'Unknown event type' };
};

export const formatDate = (date: Date) => {
	return date.toLocaleDateString('en-GB', {
		month: 'short',
		day: 'numeric',
		year: 'numeric',
		hour: 'numeric',
		minute: 'numeric',
	});
};

export const formatDuration = (duration: number) => {
	const days = Math.floor(duration / (60 * 24));
	const hours = Math.floor(
		(duration % (60 * 24)) / (60)
	);
	const minutes = duration % 60;

	if (days > 3) {
		return `${days} days`;
	}

	if (days > 1) {
		return `${days} days ${hours} hours`;
	}

	if (days > 0) {
		return `1 day ${hours} hours`;
	}

	return `${hours} hours ${minutes} mins`;
};
