import { ObservedRiverLevelSchema, ForecastRiverLevelSchema, WeatherForecastSchema, SpillEventSchema } from "@/types";
import { z } from "zod";
import useSwr, { type SWRConfiguration } from "swr";
import { storage } from "@/firebase";
import { getBytes, ref } from "firebase/storage";
import { useState, useEffect } from "react";

const createSchemaHttpFetcher =
  <T extends z.ZodTypeAny>(schema: T) =>
  async (url: string): Promise<z.infer<T>> => {
    const response = await fetch(url);
    const data = await response.json();
    console.log('HTTP fetch data', data);
    return schema.parse(data);
    };
  
const createSchemaBucketFetcher =
  <T extends z.ZodTypeAny>(schema: T) =>
  async (path: string): Promise<z.infer<T>[]> => {
    const path_ref = ref(storage, path);
    const bytes = await getBytes(path_ref);
    const data = JSON.parse(new TextDecoder().decode(bytes));
    console.log('Bucket fetch data', data);
    return schema.parse(data);
  };

const createHttpApiHook =
  <T extends z.ZodTypeAny>(url: string, schema: T, config: SWRConfiguration) =>
  () =>
      useSwr<z.infer<T>>(url, createSchemaHttpFetcher(schema), config);
  
const createBucketHook =
  <T extends z.ZodTypeAny>(path: string, schema: T, config: SWRConfiguration) =>
  () =>
      useSwr<z.infer<T>>(path, createSchemaBucketFetcher(schema), config);

export const useObservedRiverLevel = createBucketHook(
  "prediction/observation.json",
  z.array(ObservedRiverLevelSchema),
  {
    refreshInterval: 1000 * 60 * 5,
  }
);

export const useForecastRiverLevel = createBucketHook(
  "prediction/prediction.json",
  z.array(ForecastRiverLevelSchema),
  {
    refreshInterval: 1000 * 60 * 5,
  }
);

export const useWeatherForecast = createBucketHook(
  "weather/latest_forecast.json",
  z.array(WeatherForecastSchema),
  {
    refreshInterval: 1000 * 60 * 5,
  }
);

export const useSpillEvents = createHttpApiHook(
  "https://getsewageleaks-fjdislxqya-nw.a.run.app/",
  z.array(SpillEventSchema),
  {
    refreshInterval: 1000 * 60 * 15,
  }
);
