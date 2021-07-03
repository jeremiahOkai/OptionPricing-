import math
from statistics import NormalDist as nd
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import *
from pyspark.sql.functions import *
from datetime import datetime
from statistics import NormalDist as nd
import math
import os


class Preprocess:
    def __init__(self, dataFile, rfFile, divFile):
        self.dataFile = dataFile
        self.rfFile = rfFile
        self.divFile = divFile

    def transformFile(self, spark):
        data = spark.read.csv(self.dataFile, header=True, sep=',')
        rfRate = spark.read.csv(self.rfFile, header=True, sep=',')
        divRate = spark.read.csv(self.divFile, header=True, sep=';')

        # Data - processing
        for column in ['UNDLY_PRICE', 'STRIKE', 'MID']:
            data = data.withColumn(column, col(column).cast('Float'))

        dateFunc = udf(lambda x: datetime.strptime(x, '%Y%m%d'), DateType())

        data = data.withColumn('TRADE_DT', dateFunc(col('TRADE_DT'))).withColumn(
            'EXP_DT', dateFunc(col('EXP_DT'))).withColumn('POS_SIZE', lit(1)).withColumn('MULTIPLIER', lit(100))

        data = data.withColumn('DTE', datediff(data.EXP_DT, data.TRADE_DT)).withColumn(
            'dataYear', year(col('TRADE_DT')))

        data = data.filter(data.MID != 0)

        data.printSchema()

        # rfRate - processing
        for column in rfRate.columns[2:9]:
            rfRate = rfRate.withColumn(column, col(column).cast('float'))

        dateFunc1 = udf(lambda x: datetime.strptime(x, '%d.%m.%Y'), DateType())

        rfRate = rfRate.withColumn('Date', dateFunc1(col('Date')))

        rfRate = rfRate.withColumn('WeekDay', when(dayofweek('Date') == 6, 'Su')
                                   .when(dayofweek('Date') == 5, 'Sa')
                                   .when(dayofweek('Date') == 4, 'Fr')
                                   .when(dayofweek('Date') == 3, 'Th')
                                   .when(dayofweek('Date') == 2, 'We')
                                   .when(dayofweek('Date') == 1, 'Tu')
                                   .when(dayofweek('Date') == 0, 'Mo'))

        window = Window.rowsBetween(-sys.maxsize, 0)
        on = last(rfRate['ON'], ignorenulls=True).over(window)
        w1 = last(rfRate['1W'], ignorenulls=True).over(window)
        m2 = last(rfRate['2M'], ignorenulls=True).over(window)

        rfRate = rfRate.withColumn('ON', on)
        rfRate = rfRate.withColumn('1W', w1)
        rfRate = rfRate.withColumn('2M', m2)

        rfRate.printSchema()

        # divRate - processing
        dateFunc2 = udf(lambda x: datetime.strptime(
            x, '%b %d, %Y'), DateType())

        divRate = divRate.withColumn('divValue', col('Value ').cast('float'))\
                         .withColumn('Date', dateFunc2(col('Date')))\
                         .withColumn('Year', year(col('Date'))).drop('Date')

        divRate.printSchema()

        rCols = ('Date', '1M', '12M')
        rRate = rfRate.select(*rCols)

        data = data.join(rRate, rRate.Date == data.TRADE_DT, 'inner')\
                   .join(divRate, data.dataYear == divRate.Year, 'inner') \
                   .drop('Date', 'Year', 'dataYear')

        # Calculating RF_ & Div Rates
        rfCal = udf(lambda x, y, z: (x + (((z - 30) * (y - x)) / (360 - 30))) / 100
                    if x is not None and y is not None and z is not None else '', FloatType())
        divCal = udf(lambda x, y, z: ((1 + (x / y))**(z / 365) - 1)
                     if x is not None and y is not None and z is not None else '', FloatType())

        data = data.withColumn('RF_RATE', rfCal('1M', 'DTE', '12M'))\
                   .withColumn('DIV_RATE', divCal('divValue', 'UNDLY_PRICE', 'DTE')) \
                   .drop('divValue', '1M', '12M', 'Value ')

        part = Window.partitionBy(
            'TRADE_DT', 'DTE', 'TYPE').orderBy('TRADE_DT', 'DTE')
        data = data.withColumn('Part', count('DTE').over(part)) \
                   .filter(col('Part') >= 40) \
                   .drop('Part')
        return data

    @udf(returnType=FloatType())
    def newtonVol(self, UNDLY_PRICE, STRIKE, RF_RATE, DIV_RATE, DTE, MID, TYPE, sigma=0.5, max_=200, precision_=0.00001):
        if TYPE == 'C':
            for _ in range(max_):
                counter = 0
                d1 = (math.log(UNDLY_PRICE / STRIKE) + (RF_RATE - DIV_RATE + 0.5 * sigma ** 2) * (DTE / 365)) / (sigma * math.sqrt(DTE / 365))
                d2 = (math.log(UNDLY_PRICE / STRIKE) + (RF_RATE - DIV_RATE - 0.5 * sigma ** 2) * (DTE / 365)) / (sigma * math.sqrt(DTE / 365))
                vega = (1 / math.sqrt(2 * math.pi)) * UNDLY_PRICE * math.exp(-DIV_RATE * (DTE / 365)) * math.sqrt(DTE / 365) * math.exp((-nd().cdf(d1) ** 2) * 0.5)
                fx = MID - (UNDLY_PRICE * math.exp(-DIV_RATE * (DTE / 365)) * nd().cdf(d1) - STRIKE * math.exp(-RF_RATE * (DTE / 365)) * nd().cdf(d2))
                counter = -fx if fx < 0 else fx
                if counter < 0.00001:
                    return sigma
                sigma = sigma + fx / vega
            return sigma
        else:
            for _ in range(max_):
                counter = 0
                d1 = (math.log(UNDLY_PRICE / STRIKE) + (RF_RATE - DIV_RATE + 0.5 * sigma ** 2) * (DTE / 365)) / (sigma * math.sqrt(DTE / 365))
                d2 = (math.log(UNDLY_PRICE / STRIKE) + (RF_RATE - DIV_RATE - 0.5 * sigma ** 2) * (DTE / 365)) / (sigma * math.sqrt(DTE / 365))
                vega = (1 / math.sqrt(2 * math.pi)) * UNDLY_PRICE * math.exp(-DIV_RATE * (DTE / 365)) * math.sqrt(DTE / 365) * math.exp((-nd().cdf(d1) ** 2) * 0.5)
                fx = MID - (STRIKE * math.exp(-RF_RATE * (DTE / 365)) * nd().cdf(-d2) - UNDLY_PRICE * math.exp(-DIV_RATE * (DTE / 365)) * nd().cdf(-d1))
                counter = -fx if fx < 0 else fx
                if counter < 0.00001:
                    return sigma
                sigma = sigma + fx / vega
            return sigma

    @udf(returnType=FloatType())
    def euroVanillaDividend(self, UNDLY_PRICE, STRIKE, RF_RATE, DIV_RATE, DTE, IV, TYPE):
        if TYPE == 'C':
            d1 = (math.log(UNDLY_PRICE / STRIKE) + (RF_RATE - DIV_RATE + 0.5 * IV ** 2) * (DTE / 365)) / (IV * math.sqrt(DTE / 365))
            d2 = (math.log(UNDLY_PRICE / STRIKE) + (RF_RATE - DIV_RATE - 0.5 * IV ** 2) * (DTE / 365)) / (IV * math.sqrt(DTE / 365))
            fx = UNDLY_PRICE * math.exp(-DIV_RATE * (DTE / 365)) * nd().cdf(d1) - STRIKE * math.exp(-RF_RATE * (DTE / 365)) * nd().cdf(d2)
            return fx
        else:
            d1 = (math.log(UNDLY_PRICE / STRIKE) + (RF_RATE - DIV_RATE + 0.5 * IV ** 2) * (DTE / 365)) / (IV * math.sqrt(DTE / 365))
            d2 = (math.log(UNDLY_PRICE / STRIKE) + (RF_RATE - DIV_RATE - 0.5 * IV ** 2) * (DTE / 365)) / (IV * math.sqrt(DTE / 365))
            fx = STRIKE * math.exp(-RF_RATE * (DTE / 365)) * nd().cdf(-d2) - UNDLY_PRICE * math.exp(-DIV_RATE * (DTE / 365)) * nd().cdf(-d1)
            return fx

    @udf(returnType=FloatType())
    def deltaDiv(self, UNDLY_PRICE, STRIKE, RF_RATE, DIV_RATE, DTE, IV, POS_SIZE, MULTIPLIER, TYPE):
        if TYPE == 'C':
            return (nd().cdf(((( math.log(UNDLY_PRICE / STRIKE)) + (((RF_RATE -DIV_RATE) + (IV ** 2) / 2) * (DTE / 365))) / (IV * math.sqrt((DTE / 365)))))) * (POS_SIZE * MULTIPLIER)
        else:
            return ((nd().cdf((((math.log(UNDLY_PRICE / STRIKE))+(((RF_RATE - DIV_RATE) + (IV ** 2) / 2) * (DTE / 365))) / (IV * math.sqrt((DTE / 365)))))) -1) * (POS_SIZE * MULTIPLIER)

    @udf(returnType=FloatType())
    def vegaDiv(UNDLY_PRICE, STRIKE, RF_RATE, DIV_RATE, DTE, IV, POS_SIZE, MULTIPLIER):
        return ((UNDLY_PRICE * (nd().pdf((((math.log(UNDLY_PRICE / STRIKE)) + (((RF_RATE-DIV_RATE) + (IV ** 2) / 2) *(DTE/365))) / (IV*math.sqrt((DTE/365)))))) * math.sqrt((DTE/365)))/100)*(POS_SIZE*MULTIPLIER)

    @udf(returnType=FloatType())
    def thetaDiv(self, UNDLY_PRICE, STRIKE, RF_RATE, DIV_RATE, DTE, IV, POS_SIZE, MULTIPLIER, TYPE):
        if TYPE == 'C':
            theta = ((-((UNDLY_PRICE *(nd().pdf((((math.log(UNDLY_PRICE / STRIKE)) +(((RF_RATE - DIV_RATE)+(IV **2 )/2) *(DTE/365)))/(IV * math.sqrt((DTE/365)))))) * IV) /
                    (2 * math.sqrt((DTE / 365)))) -((RF_RATE-DIV_RATE)*STRIKE*(1/(math.exp((DTE/365)*(RF_RATE -DIV_RATE)))) * (nd().cdf(((((math.log(UNDLY_PRICE/STRIKE)) +
                    (((RF_RATE -DIV_RATE)+(IV**2)/2) * (DTE/365)))/(IV*math.sqrt((DTE/365)))) -(IV*math.sqrt((DTE/365))))))))/365) * (MULTIPLIER*POS_SIZE)
            return theta
        else:
            theta = ((-((UNDLY_PRICE *(nd().pdf((((math.log(UNDLY_PRICE / STRIKE)) +(((RF_RATE-DIV_RATE)+(IV ** 2) / 2) * (DTE/365))) / (IV*math.sqrt((DTE/365))))))*IV) /
                    (2*math.sqrt((DTE/365))))+((RF_RATE -DIV_RATE) * STRIKE * (1/(math.exp((DTE/365)*(RF_RATE -DIV_RATE))))*nd().cdf(-((((math.log(UNDLY_PRICE/STRIKE)) +
                    (((RF_RATE-DIV_RATE) +(IV ** 2) / 2) * (DTE/365))) / (IV*math.sqrt((DTE/365)))) -(IV*math.sqrt((DTE/365)))))))/365)*(POS_SIZE*MULTIPLIER)
            return theta

    @udf(returnType=FloatType())
    def gammaDiv(self, UNDLY_PRICE, STRIKE, RF_RATE, DIV_RATE, DTE, IV, POS_SIZE, MULTIPLIER):
        return ((nd().pdf((((math.log(UNDLY_PRICE / STRIKE)) + (((RF_RATE-DIV_RATE) +(IV ** 2) / 2) * (DTE / 365))) / (IV * math.sqrt((DTE / 365)))))) / (UNDLY_PRICE *(IV * math.sqrt((DTE/365))))) * (POS_SIZE * MULTIPLIER)

    @udf(returnType=FloatType())
    def dualDelta(self, UNDLY_PRICE, STRIKE, RF_RATE, DIV_RATE, DTE, IV, POS_SIZE, MULTIPLIER, TYPE):
        if TYPE == 'C':
            return (-math.exp(-RF_RATE*(DTE/365)) * (nd().cdf(((((math.log(UNDLY_PRICE /STRIKE)) + (((RF_RATE-DIV_RATE)+(IV**2)/2) *(DTE/365))) / (IV*math.sqrt((DTE/365)))) - (IV*math.sqrt((DTE/365))))))) *(POS_SIZE*MULTIPLIER)
        else:
            return (math.exp(-RF_RATE*(DTE/365)) * nd().cdf(-((((math.log(UNDLY_PRICE /STRIKE)) + (((RF_RATE - DIV_RATE)+ (IV**2)/2) * (DTE/365))) /(IV*math.sqrt((DTE/365)))) - (IV*math.sqrt((DTE/365))))))*(POS_SIZE*MULTIPLIER)

    @udf(returnType=FloatType())
    def charm(self, UNDLY_PRICE, STRIKE, RF_RATE, DIV_RATE, DTE, IV, POS_SIZE, MULTIPLIER):
        charm = ((DIV_RATE * math.exp(-DIV_RATE *(DTE / 365)) * nd().cdf((((math.log(UNDLY_PRICE / STRIKE)) +(((RF_RATE - DIV_RATE) + (IV**2) / 2) *(DTE / 365))) /
                (IV*math.sqrt((DTE/365))))))-(math.exp(-DIV_RATE *(DTE/365)) * (nd().pdf((((math.log(UNDLY_PRICE / STRIKE)) +(((RF_RATE -DIV_RATE) + (IV ** 2) / 2) *
                (DTE / 365))) / (IV *math.sqrt((DTE / 365))))))) * ((2 *(RF_RATE - DIV_RATE) * (DTE/365) - ((((math.log(UNDLY_PRICE / STRIKE)) + (((RF_RATE - DIV_RATE) +
                (IV ** 2) / 2) *(DTE/365))) / (IV*math.sqrt((DTE/365)))) - (IV*math.sqrt((DTE/365)))) *(IV*math.sqrt((DTE / 365)))) / (2 * (DTE / 365) * (IV * math.sqrt((DTE /365)))))) *(POS_SIZE * MULTIPLIER)

        return charm

    @udf(returnType=FloatType())
    def color(self, UNDLY_PRICE, STRIKE, RF_RATE, DIV_RATE, DTE, IV, POS_SIZE, MULTIPLIER):
        color = ((-math.exp(-DIV_RATE*(DTE/365))*(nd().pdf((((math.log(UNDLY_PRICE/STRIKE))+(((RF_RATE-DIV_RATE)+(IV**2)/2)*(DTE/365)))/(IV*math.sqrt((DTE/365)))))) /
                ((UNDLY_PRICE**2)*(IV*math.sqrt((DTE/365)))))*((2*DIV_RATE*(DTE/365))+1+((((math.log(UNDLY_PRICE/STRIKE))+(((RF_RATE-DIV_RATE)+(IV**2)/2)*(DTE/365))) /
                (IV*math.sqrt((DTE/365))))*((2*(RF_RATE-DIV_RATE)*(DTE/365)-((((math.log(UNDLY_PRICE/STRIKE))+(((RF_RATE-DIV_RATE)+(IV**2)/2)*(DTE/365)))/(IV*math.sqrt((DTE/365)))) -
                (IV*math.sqrt((DTE/365))))*(IV*math.sqrt((DTE/365))))/(IV*math.sqrt((DTE/365)))))))*(MULTIPLIER*POS_SIZE)
        return color

    @udf(returnType=FloatType())
    def vomma(self, VEGA, UNDLY_PRICE, STRIKE, RF_RATE, DIV_RATE, DTE, IV):
        vomma = VEGA*((nd().cdf((((math.log(UNDLY_PRICE / STRIKE)) + (((RF_RATE -DIV_RATE) + (IV ** 2) / 2) * (DTE /365))) / (IV * math.sqrt((DTE / 365))))) *
                (nd().cdf(((((math.log(UNDLY_PRICE / STRIKE)) + (((RF_RATE - DIV_RATE) + (IV ** 2) / 2) * (DTE / 365))) / (IV*math.sqrt((DTE/365)))) - (IV*math.sqrt((DTE/365))))))) / IV)
        return vomma

    @udf(returnType=FloatType())
    def vanna(VEGA, UNDLY_PRICE, STRIKE, RF_RATE, DIV_RATE, DTE, IV):
        return (VEGA / UNDLY_PRICE) * (1-(((math.log(UNDLY_PRICE / STRIKE)) +(((RF_RATE -DIV_RATE) + (IV ** 2) / 2) *(DTE/365))) / (IV * math.sqrt((DTE / 365)))) / (IV*math.sqrt((DTE/365))))

    @udf(returnType=FloatType())
    def decay_rate(self, THETA, FAIRVALUE):
        return (THETA/100) / FAIRVALUE

    @udf(returnType=FloatType())
    def zomma(self, GAMMA, UNDLY_PRICE, STRIKE, RF_RATE, DIV_RATE, DTE, IV):
        zomma = GAMMA*(((((math.log(UNDLY_PRICE/STRIKE))+(((RF_RATE-DIV_RATE)+(IV**2)/2) *(DTE/365)))/(IV*math.sqrt((DTE/365))))*((((math.log(UNDLY_PRICE/STRIKE))+ (((RF_RATE-DIV_RATE)+(IV**2)/2) *
                (DTE / 365))) / (IV * math.sqrt((DTE / 365)))) - (IV * math.sqrt((DTE / 365)))) - 1) / IV)
        return zomma

    @udf(returnType=FloatType())
    def wtvega(self, DTE, VEGA):
        return (math.sqrt(22 / DTE) * VEGA)

    @udf(returnType=FloatType())
    def speed(self, GAMMA, UNDLY_PRICE, STRIKE, RF_RATE, DIV_RATE, DTE, IV):
        return (-GAMMA / UNDLY_PRICE) * (1 + (((math.log( UNDLY_PRICE / STRIKE))+ (((RF_RATE - DIV_RATE) + (IV ** 2) / 2) * (DTE / 365))) / (IV * math.sqrt((DTE / 365)))) / (IV * math.sqrt((DTE / 365))))

    def calculations(self, data):
        data = data.withColumn('IV', self.newtonVol(data.UNDLY_PRICE, data.STRIKE, data.RF_RATE, data.DIV_RATE, data.DTE, data.MID, data.TYPE, lit(sigma), lit(max_), lit(precision_)))
        data = data.withColumn('IV', when(data.IV == 0, 0.0001).otherwise(data.IV))
        data = data.withColumn('FAIRVALUE', self.euroVanillaDividend(data.UNDLY_PRICE, data.STRIKE, data.RF_RATE, data.DIV_RATE, data.DTE, data.IV, data.TYPE))
        data = data.withColumn('DELTA', self.deltaDiv(data.UNDLY_PRICE, data.STRIKE, data.RF_RATE, data.DIV_RATE, data.DTE, data.IV, data.POS_SIZE, data.MULTIPLIER, data.TYPE))
        data = data.withColumn('VEGA' , self.vegaDiv(data.UNDLY_PRICE, data.STRIKE, data.RF_RATE, data.DIV_RATE, data.DTE, data.IV, data.POS_SIZE, data.MULTIPLIER))
        data = data.withColumn('THETA', self.thetaDiv(data.UNDLY_PRICE, data.STRIKE, data.RF_RATE, data.DIV_RATE, data.DTE, data.IV, data.POS_SIZE, data.MULTIPLIER, data.TYPE))
        data = data.withColumn('GAMMA', self.gammaDiv(data.UNDLY_PRICE, data.STRIKE, data.RF_RATE, data.DIV_RATE, data.DTE, data.IV, data.POS_SIZE, data.MULTIPLIER,))
        data = data.withColumn('DUAL_DELTA', self.dualDelta(data.UNDLY_PRICE, data.STRIKE, data.RF_RATE, data.DIV_RATE, data.DTE, data.IV, data.POS_SIZE, data.MULTIPLIER, data.TYPE))
        data = data.withColumn('CHARM', self.charm(data.UNDLY_PRICE, data.STRIKE, data.RF_RATE, data.DIV_RATE, data.DTE, data.IV, data.POS_SIZE, data.MULTIPLIER))
        data = data.withColumn('COLOR', self.color(data.UNDLY_PRICE, data.STRIKE, data.RF_RATE, data.DIV_RATE, data.DTE, data.IV, data.POS_SIZE, data.MULTIPLIER))
        data = data.withColumn('VOMMA', self.vomma(data.VEGA, data.UNDLY_PRICE, data.STRIKE, data.RF_RATE, data.DIV_RATE, data.DTE, data.IV))
        data = data.withColumn('VANNA', self.vanna(data.VEGA, data.UNDLY_PRICE, data.STRIKE, data.RF_RATE, data.DIV_RATE, data.DTE, data.IV))
        data = data.withColumn('DECAY_RATE', self.decay_rate(data.THETA, data.FAIRVALUE))
        data = data.withColumn('ZOMMA', self.zomma(data.GAMMA, data.UNDLY_PRICE, data.STRIKE, data.RF_RATE, data.DIV_RATE, data.DTE, data.IV))
        data = data.withColumn('WTVEGA', self.wtvega(data.DTE, data.VEGA))
        data = data.withColumn('SPEED', self.speed(data.GAMMA, data.UNDLY_PRICE, data.STRIKE, data.RF_RATE, data.DIV_RATE, data.DTE, data.IV))
        return data

    def to_postgresql(self, data, url, tableName, mode='overwrite', **properties):
        data.write.mode(mode).jdbc(url, table=tableName, properties=properties)


if __name__ == '__main__':
    url = 'jdbc:postgresql://localhost:5432/option_db'
    properties = {"user" : os.environ.get("username"), "password" : os.environ.get("pass"), "driver" : "org.postgresql.Driver"}
    spark = SparkSession.builder.master('local[*]').appName('Trading Option Mining').option('spark.jars', 'postgresql-42.2.22').getOrCreate()
    sparkInstance = Preprocess('data.csv', 'DIVRATE.csv', 'RF_RATE.csv')
    data = sparkInstance.transformFile(spark)
    data = calculation(data)
    to_postgresql(data, url, table='testtable', mode='overwrite', **properties)
