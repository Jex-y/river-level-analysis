"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { SkeletonStatusDot, StatusDot } from '@/components/ui/status-color-dot';
import { displayOrder, parameterInfo } from './config';
import { useParameters } from './hooks';

export function ParametersCard() {
  const parameters = useParameters();

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Current Conditions</CardTitle>
        <CardDescription>Status is based on DCR regulations.</CardDescription>
      </CardHeader>
      {displayOrder.map((name) => {
        const { label, formatFn, statusFn, icon: Icon } = parameterInfo[name];
        const isLoading = !parameters;

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
                  {formatFn(parameters[name])}
                </div>
              )}
              <div className="text-sm text-muted-foreground">{label}</div>
            </div>
            {isLoading ? (
              <SkeletonStatusDot />
            ) : (
              <StatusDot status={statusFn(parameters[name])} />
            )}
          </CardContent>
        );
      })}
    </Card>
  );
};
