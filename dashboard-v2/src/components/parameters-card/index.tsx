"use client";

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { SkeletonStatusDot, StatusDot } from '@/components/ui/status-color-dot';
import { Skeleton } from '@/components/ui/skeleton';
import { useParameters } from './hooks';
import { displayOrder, parameterInfo } from './config';

export function ParametersCard() {
  const parameters = useParameters();

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Conditions</CardTitle>
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
