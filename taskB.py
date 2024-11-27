import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

students = ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Hannah", "Ian", "Julia"]
math_hours = [10, 15, 20, 18, 12, 14, 11, 16, 17, 13]
science_hours = [12, 18, 17, 20, 13, 16, 12, 17, 19, 15]
english_hours = [13, 17, 18, 22, 15, 15, 15, 14, 17, 16]

math_scores = [70, 85, 90, 88, 75, 80, 72, 84, 89, 78]
science_scores = [68, 88, 85, 90, 74, 82, 73, 86, 91, 77]
english_scores = [72, 87, 83, 86, 78, 81, 75, 85, 90, 80]

time_df = pd.DataFrame({
    "Student": students,
    "Math Hours": math_hours,
    "Science Hours": science_hours,
    "English Hours": english_hours,
})

time_df.set_index('Student', inplace=True)
time_df = time_df.div(time_df.sum(axis=1), axis=0)

# print(time_df)

# Displaying normalized study hours
# time_df.plot(kind='bar', figsize=(20, 10))
# plt.title('Normalized Study Hours for Each Student')
# plt.ylabel('Normalized Hours (0-1)')
# plt.xlabel('Student')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


df = pd.DataFrame({
    "Student": students,
    "Math Hours": math_hours,
    "Science Hours": science_hours,
    "English Hours": english_hours,
    "Math Scores": math_scores,
    "Science Scores": science_scores,
    "English Scores": english_scores,
})
print(df)
df['Total_Time'] = df[['Math Hours', 'Science Hours', 'English Hours']].sum(axis=1)/3
df['Total_Score'] = df[['Math Scores', 'Science Scores', 'English Scores']].sum(axis=1)/3
correlation_matrix = df.drop(columns=['Student']).corr()
print("Correlation Matrix:")
print(correlation_matrix)
# Investigate efficient learners
df['Score_Time_Ratio'] = (df['Total_Score'] / df['Total_Time'])
print("\nScore-Time Ratio for Each Student:")
print(df[['Student', 'Score_Time_Ratio']])

# Sort students by total scores and get the top three performers
top_students = df.sort_values(by='Total_Score', ascending=False).head(3)
print("\nTop Three Performers:")
print(top_students[['Student', 'Total_Score']])

# Create a bar chart comparing the top three students' scores in each subject
top_students.set_index('Student')[['Math Scores', 'Science Scores', 'English Scores']].plot(kind='bar', figsize=(20, 10))
plt.title('Top Three Performers - Scores by Subject')
plt.ylabel('Scores')
plt.xlabel('Student')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Bonus Challenge: Performance Group Analysis
# Group students based on total score thresholds
def performance_level(score):
    if score >= 85:
        return 'High Performer'
    elif score >= 75:
        return 'Medium Performer'
    else:
        return 'Low Performer'

df['Performance_Level'] = df['Total_Score'].apply(performance_level)

# Display the distribution of performance levels
performance_counts = df['Performance_Level'].value_counts()
# Analyzing performance level distribution
print("Performance Level Distribution:")
print(performance_counts)
performance_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8), startangle=90)
plt.title('Distribution of Performance Levels')
plt.ylabel('')
plt.tight_layout()
plt.show()