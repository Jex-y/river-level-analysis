'use client';

import {
        Card,
        CardContent,
        CardDescription,
        CardFooter,
        CardHeader,
        CardTitle,
} from '@/components/ui/card';

import { LevelChart } from './chart';

export function LevelChartCard() {
        return (
                <Card>
                        <CardHeader className="flex-row items-center justify-between p-4">
                                <CardTitle>River Level</CardTitle>
                                <CardDescription>River Wear - New Elvet Bridge</CardDescription>
                        </CardHeader>
                        <CardContent className="p-2">
                                {/* If data is still loading, show empty chart */}
                                <LevelChart forecast={forecastData} observed={observedData} />
                        </CardContent>
                        <CardFooter>
                                <div className="text-muted-foreground text-xs">
                                        Predictions may be wildly inaccurate. Double check the observed data
                                        before boating!
                                </div>
                        </CardFooter>
                </Card>
        );
}
