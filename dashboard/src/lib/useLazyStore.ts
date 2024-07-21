import { useStore } from '@nanostores/react';
import type { ReadableAtom, WritableAtom } from 'nanostores';
import { useEffect, useState } from 'react';

export const useLazyStore = <T>(
	$atom: ReadableAtom<T | 'loading'> | WritableAtom<T | 'loading'>
):
	| {
			value: T;
			loading: false;
	  }
	| {
			value: undefined;
			loading: true;
	  } => {
	const atomValue = useStore($atom);
	const [componentInit, setComponentInit] = useState(false);

	useEffect(() => {
		setComponentInit(true);
	}, []);

	if (componentInit && atomValue !== 'loading') {
		return { value: atomValue, loading: false };
	}
	return { value: undefined, loading: true };
};
