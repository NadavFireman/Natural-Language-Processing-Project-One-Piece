# DATA - מקור, רישיון ושחזור

פרויקט: חיזוי Bounty בעולם One Piece (NLP + ניתוח רשתות)

מקור: One Piece Fandom Wiki דרך ה-API הציבורי - https://onepiece.fandom.com/api.php
איסוף עצמאי, בלי Kaggle או מאגר מוכן.

מה יש בקובץ onepiece_raw.csv (1,734 שורות): שם הדף (name), מזהה (pageid), קישור (url), וטקסט הדף (description) שממנו מחלצים את ה-Bounty ואת הפיצ'רים הטקסטואליים.

אחרי סינון נשארות 218 דמויות עם Bounty ידוע (משתנה המטרה), ועוד מאות דמויות ללא Bounty המשמשות לבניית הגרף.

רישיון: תוכן Fandom הוא CC BY-SA 3.0 - מותר שימוש בייחוס ושיתוף-זהה. כאן השימוש אקדמי, לא מסחרי. הנתונים הם דמויות בדיוניות, אין מידע אישי.

שחזור:
1. pip install mwparserfromhell requests pandas numpy networkx scikit-learn nltk catboost matplotlib seaborn adjustText
2. python data_collection.py  ->  מייצר onepiece_raw.csv
3. הרצת one_piece_nlp_project.ipynb

המחברת המלאה (רצה מקצה לקצה), הקוד והנתונים זמינים במאגר הפרויקט:
https://github.com/NadavFireman/One-Piece-NLP-Project

הערה: onepiece_raw.csv שוקל כ-27MB וכלול במאגר. ניתן לשחזרו במלואו מאפס עם data_collection.py (שלב 2).
