-- 创建表
CREATE TABLE house
(   id   varchar(30),
    city   varchar(30),
    local   varchar(30),
    total_price varchar(255),
    unit_price varchar(255),
    layout  varchar(30),
    area varchar(255),
    floor   varchar(255),
    time varchar(255)
);


-- 装载csv数据
LOAD DATA LOCAL INFILE 'E:/Code/PycharmProjects/Pandas/data/house.csv' INTO TABLE house 
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;