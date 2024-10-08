// import { getAnalytics } from 'firebase/analytics';
import { type FirebaseOptions, initializeApp } from 'firebase/app';
// import { getPerformance } from 'firebase/performance';
import { getStorage } from 'firebase/storage';

const firebaseConfig = {
	apiKey: 'AIzaSyDQTqkop4ZrY8rt6pmvteI-hPaekfS9Rq8',
	authDomain: 'durham-river-level.firebaseapp.com',
	projectId: 'durham-river-level',
	storageBucket: 'durham-river-level.appspot.com',
	messagingSenderId: '506521835992',
	appId: '1:506521835992:web:0bf929d49c712afdd53aad',
	measurementId: 'G-0ND87X80YC',
} satisfies FirebaseOptions;

export const app = initializeApp(firebaseConfig);
// export const analytics = getAnalytics(app);
// export const performance = getPerformance(app);
export const storage = getStorage(app);