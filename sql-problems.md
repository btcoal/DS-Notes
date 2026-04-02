# SQL Problems
## LeetCode
* https://leetcode.com/problems/rank-scores/description/
* https://leetcode.com/problems/department-top-three-salaries/description/
* https://leetcode.com/problems/consecutive-numbers/description/
* https://leetcode.com/problems/employees-earning-more-than-their-managers/description/

## Temperature Fluctuations

Write a query to find all dates with higher temperatures compared to the previous dates (yesterday.) Order dates in ascending order.

**Table: temperatures**
| Column Name | Data Type | Description          |
|-------------|-----------|----------------------|
| date        | DATE      | The date of the temperature reading |
| temperature | FLOAT     | The recorded temperature on that date |

### Solution 1
```sql
SELECT t1.date
FROM temperatures t1
JOIN temperatures t2
  ON DATEDIFF(t1.date, t2.date) = 1
WHERE t1.temperature > t2.temperature
ORDER BY t1.date ASC;
```