import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { SkeletonStatusDot, StatusDot } from '@/components/ui/status-color-dot';
import { type Parameters, parameterInfo } from '@/lib/parametersConfig';
import { useLazyStore } from '@/lib/useLazyStore';
import { currentConditionsStore } from '@/store';
import type { FC } from 'react';
import { Skeleton } from './ui/skeleton';

export type ParametersReportProps = {
	show: (keyof Parameters)[];
};

export const ParametersReport: FC<ParametersReportProps> = ({
	show,
}: ParametersReportProps) => {
	const { value: conditions, loading } = useLazyStore(currentConditionsStore);

	return (
		<Card className="w-full">
			<CardHeader>
				<CardTitle>Conditions</CardTitle>
			</CardHeader>
			{show.map((name) => {
				const { label, formatFn, statusFn, icon: Icon } = parameterInfo[name];
				const isLoading = loading || !conditions;

				return (
					<CardContent key={name} className="flex items-center gap-4">
						<div className="bg-muted rounded-full p-2 flex items-center justify-center shrink-0">
							<Icon className="w-6 h-6 text-primary" />
						</div>
						<div className="flex-1">
							{isLoading ? (
								<Skeleton className="w-40 h-10" />
							) : (
								<div className="text-4xl font-bold">
									{formatFn(conditions[name])}
								</div>
							)}
							<div className="text-sm text-muted-foreground">{label}</div>
						</div>
						{isLoading ? (
							<SkeletonStatusDot />
						) : (
							<StatusDot status={statusFn(conditions[name])} />
						)}
					</CardContent>
				);
			})}
		</Card>
	);
};
