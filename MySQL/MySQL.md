# MySQL note

## Start/Stop Server

start at window startuo or using command line `net start {server name}` in shell/command prompt , default server name is mysql80

stop with command line `net stop {server name}`

## Connect to Server

using MySQL Command Line Client or using command line `mysql [-h 127.0.0.1] [-p 3306] -u root -p`

-h: refer to the host ip, `127.0.0.1` is local ip; -h can be ignored as default is local

-p: refer to the port, `3306` is the default port when building mysql server; -p can be ignored as default is 3306

-u: refer to the sign in user

-p: refer to sign in with password, should be keyed if the user have a password

## General Rule of language

1. can be single or multiple line, end with `;`
2. can use space or indent to increase readability
3. don't differentiate upper and lower case, but keyword is suggested to use upper case
4. annotation
   * single line: --content or #content
   * multiple line /* content */

## Category of Command

| Short | Full                       | Description                                                                               |
| ----- | -------------------------- | ----------------------------------------------------------------------------------------- |
| DDL   | Data Definition Language   | Commands that define or modify the structure of database objects (tables , schemas, etc.) |
| DML   | Data Manipulation Language | Commands that manage data within tables.                                                  |
| DQL   | Data Query Language        | Commands that query data from the database.                                               |
| DCL   | Data Control Language      | Commands that control access to data                                                      |

## Data Type in MySQL

### Numeric Types

| Type                                | Size (bytes) | Signed Range                                          | Unsigned Range                                       | Description                                                                                                                                                                                                             |
| ----------------------------------- | ------------ | ----------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TINYINT`                         | 1            | [-128,127]                                            | [0,255]                                              |                                                                                                                                                                                                                         |
| `SMALLINT`                        | 2            | [-32768,32767]                                        | [0,65535]                                            |                                                                                                                                                                                                                         |
| `MEDIUMINT`                       | 3            | [-8388608,8388607]                                    | [0,16777215]                                         |                                                                                                                                                                                                                         |
| `INT`/`INTEGER`                 | 4            | [-2147483648,2147483647]                              | [0,4294967295]                                       |                                                                                                                                                                                                                         |
| `BIGINT`                          | 8            | [-2^63,2^63-1]                                        | [0,2^64-1]                                           |                                                                                                                                                                                                                         |
| `FLOAT`                           | 4            | [-3.402823466 E+38，3.402823466351 E+38]              | [1.175494351 E-38，3.402823466 E+38]                 |                                                                                                                                                                                                                         |
| `DOUBLE`                          | 8            | [-1.7976931348623157 E+308，1.7976931348623157 E+308] | [2.2250738585072014 E-308，1.7976931348623157 E+308] |                                                                                                                                                                                                                         |
| `DEC(size,d)`/`DECIMAL(size,d)` |              |                                                       |                                                      | max of size is 65, max of d is 30;<br />total number of digits is specified in  *size* ;<br />total number of digits after the decimal point is specified in the *d* parameter;<br />eg: 123.45 -- size = 5, d = 2 |

### String Types

| Type           | Size (bytes)    | Description              |
| -------------- | --------------- | ------------------------ |
| `CHAR`       | 0-255           | Fixed-length string      |
| `VARCHAR`    | 0-65535         | Variable-length string   |
| `TINYTEXT`   | 0-255           | short string text        |
| `TEXT`       | 0-65 535        | long string text         |
| `MEDIUMTEXT` | 0-16 777 215    | medium string text       |
| `LONGTEXT`   | 0-4 294 967 295 | extreme long string text |

### Date and Time Types

| Type          | Size (bytes) | Range                                     | Format              | Description                                                                                                                                                      |
| ------------- | ------------ | ----------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATE`      | 3            | 1000-01-01 ~ 9999-12-31                   | YYYY-MM-DD          | Date only                                                                                                                                                        |
| `TIME`      | 3            | -838:59:59 ~ 838:59:59                    | HH:MM:SS            | time or duration                                                                                                                                                 |
| `YEAR`      | 1            | 1901 ~2155                                | YYYY                | year                                                                                                                                                             |
| `DATETIME`  | 8            | 1000-01-01 00:00:00 ~ 9999-12-31 23:59:59 | YYYY-MM-DD HH:MM:SS | mix of date and time                                                                                                                                             |
| `TIMESTAMP` | 4            | 1970-01-01 00:00:01 ~ 2038-01-19 03:14:07 | YYYY-MM-DD HH:MM:SS | mix of date and time;<br />affected by time zone;<br />converts values from current timezone to UTC and store<br />and back to current time zone when retrieving |

### Binary Types

to store inary data (e.g., images, files)

| Type           | Size (bytes)    | Description                  |
| -------------- | --------------- | ---------------------------- |
| `TINYBLOB`   | 0-255           | binary data within 255 bytes |
| `MEDIUMBLOB` | 0-16 777 215    | medium binary data           |
| `BLOB`       | 0-65 535        | long binary data             |
| `LONGBLOB`   | 0-4 294 967 295 | extreme long binary data     |

### Other Types

* `ENUM` – Enumeration, a list of possible values
* `SET` – A set of possible values

## DDL

### Database control

#### SHOW DATABASES

`SHOW DATABASES;`  : show all databases

#### SELECT DATABASE()

`SELECT DATABASE();` : show what database is using now

#### CREATE DATABASDE

`CREATE DATABASE [ IF NOT EXISTS] {databases name} [ DEFAULT CHARSET {UTF8mb4 etc} ] [ COLLATE {sorting rule}];` : create new database, [ ] is optional function, [ IF NOT EXISTS ] - create if database not exist, can avoid error notice when database exists ; [DEFAULT CHARSET {}] - specify charset, default is utf8mb4 ; [ COLLATE {} ] - specify sorting rule, default is utf8mb4_0900_ai_ci ;

#### DROP DATABASE

`DROP DATABASE [ IF EXISTS];` :  delete database, [ IF EXISTS] - delete if database exist;

#### USE

`USE {database};` : shift to database

### Table control

#### SHOW TABLES

`SHOW TABLES;` : show all tables in this database

#### DESC

`DESC {table};` : show structure of table, similar to .info() in pandas

key column:

* **PRI** : The column is part of the Primary Key.
* **UNI** : The column has a Unique constraint (unique index).
* **MUL** : The column is part of a non-unique index (can have duplicate values), often used for columns that are part of a foreign key or just indexed for faster searches.

#### SHOW CREATE TABLE

`SHOW CREATE TABLE {table};` : show the details when creating table

#### SHOW INDEX FROM

`SHOW INDEX FROM {table};` : Show all indexes (including primary and unique keys)

#### SELECT * FROM information_schema.KEY_COLUMN_USAGE

`SELECT * FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_NAME = '{table}' AND TABLE_SCHEMA = '{databese}' AND REFERENCED_TABLE_NAME IS NOT NULL;` : show foreign keys

#### CREATE TABLE

```
CREATE TABLE { table name } (
   {Field1} {Field1Type} [ COMMENT {comment} ] [optional constraint],
   {Field2} {Field2Type} [ Comment {comment} ] [optional constraint],
   {Field3} {Field3Type} [ Comment {comment} ] [optional constraint],
   ...
) [ COMMENT {comment of table} ] ;
```

### ALTER TABLE

commands that used to modify table and its field(s)

#### ALTER  TABLE ... ADD ...

`ALTER TABLE {table} ADD {field} {type} [COMMENT {comment}] [optional constraint];` : adding new field

**Common optional constraints include:**

* `DEFAULT {value}` — Sets a default value for the new column.
* `NOT NULL` or `NULL` — Specifies whether the column can contain NULL values.
* `AFTER {existing_column}` — Specifies the position of the new column.
* `FIRST` — Adds the column as the first column in the table.
* `UNIQUE` — Adds a unique constraint.
* `PRIMARY KEY` — Makes the column a primary key, primary key is not null and unique, only one column can be defined as primary
* `CHECK()` — Limit the values that can be placed in a column, ensuring data integrity by specifying a condition that each row must satisfy
* `FOREIGN KEY ({key_field}) REFRENCES {foreign_table({key_field})} [ON DELETE {action}] [ON UPDATE {action}]` — Link two tables together, ensuring that the value in one table (the child table) matches a value in another table (the parent table), maintaining referential integrity, `[ON DELETE {action}]` and `[ON UPDATE {action}]` decide what to do if the parent row delete/upadate:
  * `RESTRICT` : Prevents the action if there are related rows in the child table (default in MySQL if ON is not specified).
  * `NO ACTION` : Similar to `RESTRICT`; the action is rejected if it would break referential integrity.
  * `SET NULL` : Sets the foreign key column in the child table to `NULL` if the referenced parent row is deleted or updated.
  * `SET DEFAULT` : Sets the foreign key column to its default value (rarely used; requires the column to have a default).
  * `CASCADE` : Propagates the delete or update to child rows.

#### ALTER TABLE ... MODIFY ...

`ALTER TABLE {table} MODIFY {field} {new type} [constraint];` : modify the type and/or constraint of field in table

#### ALTER TABLE ... CHANGE ...

`ALTER TABLE {table} CHANGE {origin field} {new field} {type} [ COMMENT {comment}] [constraint];` : change field name ,type , comment and constraint

#### ALTER TABLE ... DROP...

`ALTER TABLE {table} DROP {field}/{constraint};` : delete field or constraint

#### ALTER TABLE ... RENAME ...

`ALTER TABLE {table} RENAME TO {new table name};`: rename the table

#### ALTER TABLE ... ADD CONSTRAINT ...

`ALTER TABLE {table} ADD CONSTRAINT [{contraint name}] {constraint key} {field};` : add constraint to the field, constraint name can be named manually or automatically if inorged

add primary key: `ALTER TABLE {table} ADD CONSTRAINT [{constraint name}] PRIMARY KEY ({field})`

add foreign key: `ALTER TABLE {table} ADD CONSTRAINT [{constraint name}] FOREIGN KEY ({field}) REFERENCES {target table}({target field});`

add unique constraint : `ALTER TABLE {table} ADD CONSTRAINT [{constraint name}] UNIQUE ({field});`

add check constraint : `ALTER TABLE {table} ADD CONSTRAINT [{constraint name}] CHECK ({field} {condition});`

### DROP TABLE

`DROP TABLE [ IF EXISTS ] {table};` : delete table

### TRUNCATE TABLE

`TRUNCATE TABLE {table};` : (format) delete and recreate table

### AUTO_INCREMENT

a key used to automatically generate a unique integer value for a field, when insert new row, MySQL will automatically assign the next available integer value for this field

* Only one `AUTO_INCREMENT` column per table.
* The column must be defined as a key (usually `PRIMARY KEY`).

usage in `CREATE`:

```
CREATE TABLE users (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(50)
);
```

usage in `ALTER`:

`ALTER TABLE {table} AUTO_INCREMENT = {start};` : `AUTO_INCREMENT` will automatically applied to the primary field and the next value will start from {start}

## DML

### INSERT INTO

`INSERT INTO {table} (field1,field2,field3,...) VALUES (value1,value2,value3,...);` : insert new row within specific fields into table

`INSERT INTO {table} VALUES (value1,value2,value3,...);` : insert new row with all fields into table

`INSERT INTO {table} (field1,field2,field3,...) VALUES (value1,value2,value3,...),(value1,value2,value3,...),...;` : insert multiple rows  within specific fields into table

`INSERT INTO {table} VALUES (value1,value2,value3,...),(value1,value2,value3,...),...;` : insert multiple rows with all fields into table

### UPDATE ... SET ...

`UPDATE {table} SET {field1} = {value}, {field2} = {value}, ... [WHERE {condition}];` : change the value of field depend on condition, without condition, change all value in the field, eg: `UPADAT test SET name = 'Alex' WHERE id = 123` change the name to Alex at row which id =123; update value to  `null` can delete value in specific cell

### DELETE FROM

`DELETE FROM {table} [WHERE {condition}];` : delete row depend on condition, without condition, delete all rows in table

## DQL

`SELECT [DISTINCT] {} [AS alias] FROM {} [WHERE {}] [GRUOP BY {} HAVING {}] [ORDER BY {}] [LIMIT {}];`

### SELECT ... FROM

`SELECT {field1,field2,... or *} FROM {table}` : Query specific fields or `*` for all fields in table

`SELECT {field1 [AS alias1],field2 [AS alias2],...} FROM {table}` : Give a alias to the field when querying, as can be omitted

`SELECT DISTINCT {field1,field2,... or * } FROM {table}` : Ignore duplicates when querying

### Query with condition

`SELECT {field1,field2,... or *} FROM {table} WHERE {condition}` : query with condition, eg: `WHERE age > 35`

Comparison Operator:

| Operator                | Description                                                                 | Example                               |
| ----------------------- | --------------------------------------------------------------------------- | ------------------------------------- |
| =                       | Equal to                                                                    | `age = 18`                          |
| <> or !=                | Not equal to                                                                | `age <> 18`                         |
| >                       | Greater than                                                                | `age > 18`                          |
| <                       | Less than                                                                   | `age < 18`                          |
| >=                      | Greater than or equal to                                                    | `age >= 18`                         |
| <=                      | Less than or equal to                                                       | `age <= 18`                         |
| BETWEEN {min} AND {max} | Between two values (inclusive)                                              | `age BETWEEN 18 AND 30`             |
| LIKE                    | Pattern matching,`_`match any single char, `%` match any length of char | `name LIKE 'A%', name start with A` |
| IN                      | Match any value in a list                                                   | `age IN (18, 21)`                   |
| IS NULL                 | Is NULL                                                                     | `age IS NULL`                       |
| IS NOT NULL             | Is not NULL                                                                 | `age IS NOT NULL`                   |

### Aggregation

`SELECT {field} {agg func} FROM {table}` : aggregate with field and use agg func to calculate count, sum, ....; if field not provide, aggregate the whole table; eg: `SELECT department COUNT(*) FROM employer;` aggregate by department and count the row of each group; eg: `SELECT COUNT(*) FROM employer;` count rows in employer;

Aggregation Function:

| Function           | Description                       | Example usage          |
| ------------------ | --------------------------------- | ---------------------- |
| `COUNT()`        | Counts rows, null is not included | `COUNT(*)`           |
| `SUM()`          | Sums values in a column           | `SUM(salary)`        |
| `AVG()`          | Calculates the average value      | `AVG(score)`         |
| `MIN()`          | Finds the minimum value           | `MIN(age)`           |
| `MAX()`          | Finds the maximum value           | `MAX(price)`         |
| `GROUP_CONCAT()` | Concatenates values into a string | `GROUP_CONCAT(name)` |

`SELECT {grouping field} {agg func1, agg func2,...} FROM {table} [WHERE {condition}] GROUP BY {grouping field} [HAVING {filter condition}];` : field by appointed field, `[HAVING {}]`filter with condition after grouping, `select department ,AVG(age) , AVG(salary), COUNT(*) from employee group by department HAVING COUNT(*)>=4;`

### Sorting

 `SELECT {field1,field2,... or *} FROM {table} ORDER BY {field1} {sort by}, {field2} {sort by},...;` : display result sort by `ASC`/`DESC`

### Paging

 `SELECT {field1,field2,... or *} FROM {table} LIMIT {started index}, {row in each page};` : limit the number of rows display, started index = (page number -1) * (row in each page), eg first page (show 0-9) = LIMIT 0, 10; second page (show 10-19) = LIMIT 10, 10; second page but start with 11 (show 11-19) = LIMIT 11,19

execute order : `FROM` > `WHERE` > `GROUP BY` >  `HAVING` > `SELECT` > `ORDER BY` > `LIMIT`

### Union

`SELECT {field list} FROM ... UNION [ALL] SELECT {field list};` : `UNION` can join one or more `SELECT` query result together, {field list} must be same in each query. `UNION ALL` will combine all row even there are duplicates between query result, while `UNION` will remove duplicates

### Multiple tables query

#### Cross join

`SELECT {fields list} FROM {table list};` : Returns the Cartesian product of all tables ( N\*N\*N... rows )

#### Inner join

`SEELECT {fields list} FROM {table list} WHERE {join condition};` : Implicit inner join, `{join condition}`should be replace with join of all {child foreign key field} = {parent reference field}, eg: `{foreign key field 1} = {reference field 1} AND {foreign key field 2} = {reference field 2} AND ...`

`SELECT {fields list} FROM {table 1} [INNER] JOIN {table 2} ON {condition 1} [INNER] JOIN {table 3} ON {condition 2} ...;` : Explicit inner join, inner can be omit, eg: `select dept.id, dept.name,emp.id, emp.name, emp.age, emp.salary, emp.enterdate, emp.managerid, emp.dept_id, salgrade.grade from dept inner join emp on dept.id = emp.dept_id inner join salgrade on emp.salary BETWEEN salgrade.losal AND salgrade.hisal;`

#### Left outer join

`SELECT {field list} FROM {table1} LEFT [OUTER] JOIN {table2} ON {condition};`

#### Right outer join

`SELECT {field list} FROM {table1} RIGHT [OUTER] JOIN {table2} ON {condition};`

eg:

```
select dept.id, dept.name,emp.id, emp.name, emp.age, emp.salary, emp.enterdate, emp.managerid, emp.dept_id, salgrade.grade from emp
    left join dept on dept.id = emp.dept_id
    inner join salgrade on emp.salary BETWEEN salgrade.losal AND salgrade.hisal;
```

#### Self join

`SELECT {field list} FROM {table1} [INNER/LEFT [OUTER]/RIGHT [OUTER]] JOIN {table1} on {condition};` : can be used when one field is refer to another field in same table, eg:

```
select e1.id, e1.name, e1.age, e1.salary, e1.enterdate, e1.managerid,e2.name as managename, e1.dept_id from emp as e1
    left join emp as e2 on e2.id = e1.managerid;
```

#### Full outer join

MySQL doesn't support FULL OUTER JOIN directly, but can be simulated with `UNION`, full outer join can prevent row removed due to disappear in other tables but no return Cartesian product

```
SELECT {field list} FROM {table1} LEFT [OUTER] JOIN {table2} ON {condition}
UNION
SELECT {field list} FROM {table1} RIGHT [OUTER] JOIN {table2} ON {ocndition};
```

### Subqueries

use one query result in other query condition

#### Scalar subquery

the query result used in condition is a single value

`SELECT {field list} FROM {table} WHERE {field} {=/<>/>/>=/</<=} ({subquery});` : `{subquery}` should return a single value

#### Column subquery

`SELECT {field list} FROM {table} WHERE {field} {IN/NOT IN/ANY/SOME/ALL} ({subquery});` : `{subquery}` should return a column of values, use to select row in or not in subquery (`IN/NOT IN`), or compare `(=/<>/>/>=/</<=)`with all/any value in subquery (`ANY/SOME/ALL`)

#### Row subquery

`SELECT {field list} FROM {table} WHERE {field} {=/<>/>/>=/</<=} ({subquery});` : `{subquery}` should return a row of values, similar to scalar subquery, but compare with multiple values

#### Table subquery

`SELECT {field list} FROM {table} WHERE {field} {IN/NOT IN/ANY/SOME/ALL} ({subquery});`: `{subquery}` should return a table of values, similar to column subquery, but compare with multiple columns

## DCL

### User management

#### List all users

```
USE mysql;
SELECT * FROM user;
```

user data is store in mysql.user

* Host: The host from which the user can connect (e.g., 'localhost', '%').

  * localhost -- the user can connect only from the same machine where the MySQL server is running.
  * % -- the user can connect from any host (wildcard).
  * An IP address (e.g., 192.168.1.100)  -- the user can connect only from that specific IP.
  * A domain name (e.g., myhost.example.com) -- the user can connect only from that host.
* User: The username.
* authentication_string or Password: The user's password hash (field name depends on MySQL version).
* plugin: The authentication plugin used (e.g., 'mysql_native_password').
* Select_priv, Insert_priv, Update_priv, etc.: Columns indicating specific privileges (Y/N) for the user.
* account_locked: Indicates if the account is locked (in newer versions).
* ssl_type, ssl_cipher, x509_issuer, x509_subject: SSL-related fields for secure connections.
* Select_priv: Allows SELECT (read) queries on all databases.
* Insert_priv: Allows INSERT (add data) queries.
* Update_priv: Allows UPDATE (modify data) queries.
* Delete_priv: Allows DELETE (remove data) queries.
* Create_priv: Allows creating new databases and tables.
* Drop_priv: Allows dropping (deleting) databases and tables.
* Reload_priv: Allows reloading server settings (e.g., FLUSH commands).
* Shutdown_priv: Allows shutting down the MySQL server.
* Process_priv: Allows viewing all processes with SHOW PROCESSLIST.
* File_priv: Allows reading and writing files on the server host.
* Grant_priv: Allows granting privileges to other users.
* References_priv: Allows creating foreign keys.
* Index_priv: Allows creating and dropping indexes.
* Alter_priv: Allows altering tables (e.g., adding columns).
* Show_db_priv: Allows seeing all databases with SHOW DATABASES.
* Super_priv: Allows many administrative operations (e.g., KILL, SET GLOBAL).
* Create_tmp_table_priv: Allows creating temporary tables.
* Lock_tables_priv: Allows locking tables for access control.
* Execute_priv: Allows executing stored routines.
* Repl_slave_priv: Allows the user to read binary log events for replication.
* Repl_client_priv: Allows the user to ask where master or slave servers are.
* Create_view_priv: Allows creating views.
* Show_view_priv: Allows viewing view definitions.
* Create_routine_priv: Allows creating stored procedures and functions.
* Alter_routine_priv: Allows altering or dropping stored routines.
* Create_user_priv: Allows creating, dropping, and renaming user accounts.
* Event_priv: Allows creating, altering, dropping, and viewing events.
* Trigger_priv: Allows creating and dropping triggers.

#### CREATE USER

`CREATE USER '{user name}'@'{host}' [IDENTIFIED BY {password}];` : create user, optional encrypt with `IDENTIFIED BY{}`

#### RENAME USER ... TO ...

`RENAME USER '{user name}'@'host' TO '{new name}'@'new host';` : change user name and/or host

#### ALTER USER

##### Change password

`ALTER USER '{user name}'@{host}' IDENTIFIED BY {new password};` : change user password with default authentication plugin (< MySQL 8.0 : mysql_native_password , >= MySQL 8.0 : caching_sha2_password)

`ALTER USER '{user name}'@'{host}' IDENTIFIED WITH {plugin name} BY {new password} ;` : change user password with specific authentication plugin

* mysql_native_password: The traditional password-based authentication.
* caching_sha2_password: The default in MySQL 8.0, uses SHA-256 hashing.
* sha256_password: Uses SHA-256 for password hashing.
* auth_socket: Authenticates users via the Unix socket file (no password).

##### Change authentiction plugin

##### Lock or unlock user account

##### Set password expiration

##### Require SSL or other connection

#### DROP USER

`DROP USER 'user name'@'host';` : delete user

### Privilege management

#### SHOW GRANTS FOR ...

`SHOIW GRANTS FOR '{user name}'@'{host}';` : show privileges of the user. If only USAGE shown in privilege list, the user can only login to the server

#### GRANT ... ON ... TO ...

`GRANT {privileges list} ON {database}.{table} TO '{user name}'@'{host}';` : grant privileges to the user, eg: `GRANT SELECT, INSERT, UPDATE ON *.* TO 'test'@'localhost' ;`

{privileges list} can be filled with ALL for all privileges

{database}.{table} can filled with \*.\* for all databases and tables

| Privilege    | Description                               |
| ------------ | ----------------------------------------- |
| SELECT       | Allows reading data from tables           |
| INSERT       | Allows adding new rows to tables          |
| UPDATE       | Allows modifying existing data in tables  |
| DELETE       | Allows removing rows from tables          |
| CREATE       | Allows creating new databases and tables  |
| DROP         | Allows deleting databases and tables      |
| ALTER        | Allows modifying table structure          |
| INDEX        | Allows creating and dropping indexes      |
| GRANT OPTION | Allows granting privileges to other users |
| REFERENCES   | Allows creating foreign key constraints   |
| EXECUTE      | Allows executing stored routines          |
| SHOW VIEW    | Allows viewing the definition of views    |
| CREATE VIEW  | Allows creating new views                 |
| TRIGGER      | Allows creating and dropping triggers     |

#### REVOKE ... ON ... FROM ...

`REVOKE {privilege list} ON {datbase}.{table} FROM '{user name}'@'{host}';` : remove privileges from the user

## Summary

### DDL

| Command                                                                                                                                                   | Description                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `SHOW DATABASES;`                                                                                                                                       | show all databases                                |
| `SELECT DATABASE();`                                                                                                                                    | show what database is using now                   |
| `CREATE DATABASE [ IF NOT EXISTS] {databases name} [ DEFAULT CHARSET {UTF8mb4 etc} ] [ COLLATE {sorting rule}];`                                        | creaate database                                  |
| `DROP DATABASE [ IF EXISTS];`                                                                                                                           | delete database                                   |
| `USE {database};`                                                                                                                                       | use database                                      |
| `SHOW TABLES;`                                                                                                                                          | list all table                                    |
| `DESC {table};`                                                                                                                                         | show structure of table                           |
| `SHOW CREATE TABLE {table};`                                                                                                                            | show the details when creating table              |
| `SHOW INDEX FROM {table};`                                                                                                                              | show index(include primary and unique keys)       |
| `SELECT * FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_NAME = '{table}' AND TABLE_SCHEMA = '{databese}' AND REFERENCED_TABLE_NAME IS NOT NULL;` | show foreign keys                                 |
| `CREATE TABLE { table name } ( {field}{field type} [COMMENT {comment}] [optional constraint]) [COMMENT {comment of table}] ;`                           | create a table                                    |
| `ALTER TABLE {table} ADD {field} {type} [COMMENT {comment}] [optional constraint];`                                                                     | add new field                                     |
| `ALTER TABLE {table} MODIFY {field} {new type};`                                                                                                        | modify the type of field                          |
| `ALTER TABLE {table} CHANGE {origin field} {new field} {type} [ COMMENT {comment}] [constraint];`                                                       | change field name ,type , comment and constraint |
| `ALTER TABLE {table} DROP {field}/{constraint};`                                                                                                        | delete field or constraint                        |
| `ALTER TABLE {table} RENAME TO {new table name};`                                                                                                       | rename the table                                  |
| `ALTER TABLE {table} ADD CONSTRAINT [{contraint name}] {constraint key} {field};`                                                                       | add constraint                                    |
| `DROP TABLE [ IF EXISTS ] {table};`                                                                                                                     | delete table                                      |
| `TRUNCATE TABLE {table};`                                                                                                                               | (format) delete and recreate table                |

### DML

| Command                                                                                                              | Description                                             |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `INSERT INTO {table} (field1,field2,field3,...) VALUES (value1,value2,value3,...);`                                | insert new row within specific fields into table        |
| `INSERT INTO {table} VALUES (value1,value2,value3,...);`                                                           | insert new row with all fields into table               |
| `INSERT INTO {table} (field1,field2,field3,...) VALUES (value1,value2,value3,...),(value1,value2,value3,...),...;` | insert multiple rows  within specific fields into table |
| `INSERT INTO {table} VALUES (value1,value2,value3,...),(value1,value2,value3,...),...;`                            | insert multiple rows with all fields into               |
| `UPDATE {table} SET {field1} = {value}, {field2} = {value}, ... [WHERE {condition}];`                              | change the value of field                               |
| `DELETE FROM {table} [WHERE {condition}];`                                                                         | delete row                                              |

### DQL

`SELECT [DISTINCT] {field} [AS alias] FROM {} [WHERE {}] [ORDER BY {}] [LIMIT {}];`

`SELECT {field} {agg function} FROM {table} [WHERE {condition}] [GRUOP BY {} HAVING {}];`

| Function           | Description                       |
| ------------------ | --------------------------------- |
| `COUNT()`        | Counts rows, null is not included |
| `SUM()`          | Sums values in a column           |
| `AVG()`          | Calculates the average value      |
| `MIN()`          | Finds the minimum value           |
| `MAX()`          | Finds the maximum value           |
| `GROUP_CONCAT()` | Concatenates values into a string |

### DCL

| Command                                                                                 | Description                     |
| --------------------------------------------------------------------------------------- | ------------------------------- |
| `USE mysql;` <br /> `SELECT * FROM user;`                                          | list all user                   |
| `CREATE USER '{user name}'@'{host}' [IDENTIFIED BY {password}] ;`                     | create user                     |
| `RENAME USER '{user name}'@'host' TO '{new name}'@'new host';`                        | change user name and/or host    |
| `ALTER USER '{user name}'@{host}' IDENTIFIED [WITH {plugin name}] BY {new password};` | change user password            |
| `DROP USER 'user name'@'host';`                                                       | delete user                     |
| `SHOIW GRANTS FOR '{user name}'@'{host}';`                                            | show privileges of the user     |
| `GRANT {privileges list} ON {database}.{table} TO '{user name}'@'{host}';`            | grant privileges to the user    |
| `REVOKE {privilege list} ON {datbase}.{table} FROM '{user name}'@'{host}';`           | remove privileges from the user |
| `UNION [ALL]`                                                                         | combine two queries' returns    |
| `[INNER] JOIN ... ON ...`                                                             | inner join                      |
| `LEFT [OUTER] JOIN ... ON ...`                                                        | left outer join                 |
| `RIGHT [OUTER] JOIN ... ON ...`                                                       | right outer join                |

## Function

### String function

| Function                     | Description                                                               | Example usage                               |
| ---------------------------- | ------------------------------------------------------------------------- | ------------------------------------------- |
| `CONCAT(S1,S2,...,Sn)`     | Concatenates two or more strings                                          | `CONCAT('Hello', ' ', 'World')` ->       |
| `LENGTH(str)`              | Returns the length of a string (bytes)                                    | `LENGTH('abc')`                           |
| `CHAR_LENGTH(str)`         | Returns the number of characters                                          | `CHAR_LENGTH('abc')` -> 3                 |
| `LOWER(str)`               | Converts to lowercase                                                     | `LOWER('ABC')` -> 'abc'                   |
| `UPPER(str)`               | Converts to uppercase                                                     | `UPPER('abc')` -> 'ABC'                   |
| `SUBSTRING(str,start,len)` | Extracts a substring of `str`, from `start`to `start `+`len`      | `SUBSTRING('abcdef', 1, 3)` -> 'abc'     |
| `LEFT(str,len)`            | Leftmost characters                                                       | `LEFT('abcdef', 3)` -> 'abc'              |
| `RIGHT(str,len)`           | Rightmost characters                                                      | `RIGHT('abcdef', 2)` -> 'ef'              |
| `LPAD(str,n,pad)`          | Fill `str`  with `pad` on its left to make the char_length = `n`  | `LPAD('abc',5,'_')` -> '__abc'            |
| `RPAD(str,n,pad)`          | Fill `str`  with `pad` on its right to make the char_length = `n` | `RPAD('abc',5,'_')` -> 'abc__'            |
| `TRIM(str)`                | Removes leading/trailing spaces                                           | `TRIM('  hello  ')` -> 'hello'            |
| `REPLACE(str,org,rpl)`     | Replaces occurrences of a substring                                       | `REPLACE('abcabc', 'a', 'x')` -> 'xbcxbc' |
| `INSTR()`                  | Returns position of substring                                             | `INSTR('hello', 'e')` -> 2                |
| `REVERSE()`                | Reverses a string                                                         | `REVERSE('abc')` -> 'cba'                 |

### Numerical function

| Function                | Description                                                                   | Example usage                       |
| ----------------------- | ----------------------------------------------------------------------------- | ----------------------------------- |
| `ABS(x)`              | Absolute value                                                                | `ABS(-5)` → 5                    |
| `CEIL(x)`             | Smallest integer ≥ x (ceiling)                                               | `CEIL(2.3)` → 3                  |
| `FLOOR(x)`            | Largest integer ≤ x (floor)                                                  | `FLOOR(2.8)` → 2                 |
| `ROUND(x, d)`         | Rounds x to d decimal places                                                  | `ROUND(2.678, 2)` → 2.68         |
| `MOD(x, y)`           | Remainder of x divided by y                                                   | `MOD(10, 3)` → 1                 |
| `POWER(x, y)`         | x raised to the power y                                                       | `POWER(2, 3)` → 8                |
| `SQRT(x)`             | Square root                                                                   | `SQRT(16)` → 4                   |
| `RAND()`              | Random floating-point value (0 ≤ x < 1)                                      | `RAND()*10` random number in 0-10 |
| `SIGN(x)`             | Returns `1` if x is positive,  `0` if x is zero,  `-1` if x is negative | `SIGN(-10)` → -1                 |
| `GREATEST(a, b, ...)` | Largest value                                                                 | `GREATEST(2, 5, 3)` → 5          |
| `LEAST(a, b, ...)`    | Smallest value                                                                | `LEAST(2, 5, 3)` → 2             |

### Date and time function

| Function                            | Description                          | Example usage                                |
| ----------------------------------- | ------------------------------------ | -------------------------------------------- |
| `NOW()`                           | Current date and time                | `NOW()`                                    |
| `CURDATE()`                       | Current date (YYYY-MM-DD)            | `CURDATE()`                                |
| `CURTIME()`                       | Current time (HH:MM:SS)              | `CURTIME()`                                |
| `DATE()`                          | Extracts date part from a datetime   | `DATE('2024-06-25 10:20:30')`              |
| `YEAR(date)`                      | Year part of a date                  | `YEAR('2024-06-25')`                       |
| `MONTH(date)`                     | Month part of a date                 | `MONTH('2024-06-25')`                      |
| `DAY(date)`                       | Day part of a date                   | `DAY('2024-06-25')`                        |
| `HOUR(time)`                      | Hour part of a time                  | `HOUR('10:20:30')`                         |
| `MINUTE(time)`                    | Minute part of a time                | `MINUTE('10:20:30')`                       |
| `SECOND(time)`                    | Second part of a time                | `SECOND('10:20:30')`                       |
| `DATE_ADD(date, INTERVAL n unit)` | Adds interval to date                | `DATE_ADD('2024-06-25', INTERVAL 1 DAY)`   |
| `DATE_SUB(date, INTERVAL n unit)` | Subtracts interval from date         | `DATE_SUB('2024-06-25', INTERVAL 1 MONTH)` |
| `DATEDIFF(date1, date2)`          | Difference in days between two dates | `DATEDIFF('2024-06-25', '2024-06-20')`     |
| `STR_TO_DATE(str, format)`        | Parses string to date using format   | `STR_TO_DATE('25-06-2024', '%d-%m-%Y')`    |
| `DATE_FORMAT(date, format)`       | Formats date as a string             | `DATE_FORMAT('2024-06-25', '%d/%m/%Y')`    |

### Control flow function

| Function                                    | Description                                                  | Example usage                                                                                                                                                |
| ------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `IF(expr, true_val, false_val)`           | Returns `true_val` if `expr` is true, else `false_val` | `IF(score >= 60, 'Pass', 'Fail')`                                                                                                                          |
| `IFNULL(expr1, expr2)`                    | Returns `expr1` if not NULL, else returns `expr2`        | `IFNULL(name, 'Anonymous')`                                                                                                                                |
| `NULLIF(expr1, expr2)`                    | Returns NULL if `expr1 = expr2`, else returns `expr1`    | `NULLIF(a, b)`                                                                                                                                             |
| `CASE ... WHEN ... THEN ... ELSE ... END` | Multi-branch conditional logic                               | `CASE WHEN score >= 90 THEN 'A' WHEN score >= 60 THEN 'B' ELSE 'C' END`<br />`CASE address WHEN 'kl' THEN 'local' WHEN 'foreign' THEN 'foreign' ELSE '` |

## Transactions

### ACID Properties

* **Atomicity** : All operations succeed or all fail
* **Consistency** : Database remains in a valid state
* **Isolation** : Transactions don't interfere with each other
* **Durability** : Committed changes are permanent

### Starting and ending transactions

`START TRANSATION;` : after executed, new transactions will be create, all commands after it won't commit automically, untill `COMMIT;` or `ROLLBACK;` executed

`SAVEPOINT {savepoint name};` : create a savepoint for partial rollback

`COMMIT;` : save all changes

`ROLLBACK;` : undo all changes

`ROLLBACK TO {savepoint name};` : rollback to specific savepoint

### Set aouto-commit mode

`SELECT @@autocommit;` : check autocommit mode of the system. If 1, aoutocommit mode on, changes will be saved automically after each commands executed; if 0, autocommit mode off, changes won't be save without `commit;`

`SET @@autocommit = 0;` :  Disable auto-commit, turn all commands execute in transaction, all commands executed will be count as one transaction untill a `COMMIT;` or `ROLLBACK;` executed

`SET @@autocommit = 1;` :  Enable auto-commit

### Concurrent transactions

#### Problems

1. Dirty read: A transaction reads uncommitted data from another transaction.
2. Non-repeatable read: A transaction reads the same data twice but gets different results.
3. Phantom read: A transaction sees new rows that weren't there in a previous read.
4. Lost update: Two transactions update the same data, and one overwrites the other's changes.

#### Isolation levels

`SELECT @@transaction_isolation;`: check current isolation level

`SET [SESSION/GLOBAL] TRANSACTION ISOLATION LEVEL {isolation level};` : set isolation level, `SESSION` only affects all transactions in the current session/connection; `GLOBAL` affects all new connections to the server; if affect level no specified, isolation level change only affect next transaction.

| Level                | Description                                          | Dirty read | Non-repeatable read | Phantom read | Lost update |
| -------------------- | ---------------------------------------------------- | ---------- | ------------------- | ------------ | ----------- |
| `READ UNCOMMITTED` | Can read uncommitted changes from other transactions | ✓         | ✓                  | ✓           | ✓          |
| `READ COMMITTED`   | Only reads committed data                            | ✕         | ✓                  | ✓           | ✓          |
| `REPEATABLE READ`  | Consistent reads within the transaction              | ✕         | ✕                  | ✓           | ✓          |
| `SERIALIZABLE`     | Transactions execute as if they were sequential      | ✕         | ✕                  | ✕           | ✕          |
