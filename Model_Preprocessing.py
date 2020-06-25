from sklearn.base import BaseEstimator, TransformerMixin 
import pandas as pd
import joblib
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array

#Custom Transformer that extracts columns passed as argument to its constructor x`
class FeatureSelector( BaseEstimator,TransformerMixin ):

    def __init__( self, feature_names = None  ):
        self._feature_names = feature_names 
     
    def fit( self, X, y = None , **fit_params ):
        return self 

    def transform( self, X, y = None ):
        return X[ self._feature_names ]
    
#Custom Transformer that encodes categorical variables   

class Encoding( BaseEstimator,TransformerMixin):
    def __init__(self , categorical_columns = None ):
        import pandas as pd
        self.col = categorical_columns
    
    def transform( self , data ):
        import pandas as pd
        
        self.data = pd.DataFrame( data , columns = self.col ) 
        part_of_day_mapping = {'Early Morning' : 1 , 'Evening' : 2 , 'Morning' : 3 , 
                                   'Noon' : 4 , 'Late Night' : 5 , 'Night' : 6 }
        Side_mapping = { 'R' : 1 , 'L' : 0 }

        self.data['City'] = self.data['City'].apply( lambda x : self.City_mapping.get(x, 0 ) )
        self.data['County'] = self.data['County'].apply( lambda x : self.County_mapping.get(x, 0 ) )
        self.data['State'] = self.data['State'].apply( lambda x : self.State_mapping.get(x, 0 ) )
        self.data['Street'] = self.data['Street'].apply( lambda x : self.Street_mapping.get(x, 0 ) )
        self.data['Timezone'] = self.data['Timezone'].apply( lambda x : self.Timezone_mapping.get(x, 0 ) )
        self.data['Airport_Code'] = self.data['Airport_Code'].apply( lambda x : self.Airport_Code_mapping.get(x, 0 ) )
        self.data['Wind_Direction'] = self.data['Wind_Direction'].apply( lambda x : self.Wind_Direction_mapping.get(x, 0 ) )
        self.data['Weather_Condition'] = self.data['Weather_Condition'].apply( lambda x : self.Weather_Condition_mapping.get(x, 0 ) )
        self.data['part_of_day'] = self.data['part_of_day'].apply( lambda x : part_of_day_mapping.get(x, 0 ) )
        self.data['Side'] = self.data['Side'].apply( lambda x : Side_mapping.get(x, 0 ) )
        
        for col in ['Amenity','Bump','Crossing','Give_Way','Junction','Railway','Roundabout','Station','Stop','Traffic_Signal','Turning_Loop']:
            self.data[col] = self.data[col].astype(int)
   
        if 'Severity' in self.data.columns:
            return self.data.drop( 'Severity' , axis = 1 )
        else:
            return self.data
    
    def fit(self, data, y=None, **fit_params):
        
        import pandas as pd
        data = pd.DataFrame( data , columns = self.col )
        data['Severity'] = data['Severity'].astype('int')
        self.Airport_Code_mapping = data.groupby('Airport_Code')['Severity'].mean().to_dict()
        self.Timezone_mapping = data.groupby('Timezone')['Severity'].count().to_dict()
        self.City_mapping = data.groupby('City')['Severity'].mean().to_dict()
        self.County_mapping = data.groupby('County')['Severity'].mean().to_dict()
        self.State_mapping = data.groupby('State')['Severity'].mean().to_dict()
        self.Street_mapping = data.groupby('Street')['Severity'].mean().to_dict()
        self.Wind_Direction_mapping =data.groupby('Wind_Direction')['Severity'].count().to_dict()
        self.Weather_Condition_mapping = data.groupby('Weather_Condition')['Severity'].count().to_dict()
        
        return self
    
class CustomeScaler(MinMaxScaler):
    def transform(self, X):
#         X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
#                         force_all_finite="allow-nan")
        X *= self.scale_
        X += self.min_
        return X
    
class PreProcessing:
    
    def __init__(self):
        pass
    
    def train_pipe(self):

        f = FeatureSelector()
        e = Encoding()

        data = pd.read_excel( 'C:\\Users\\Administrator\\Desktop\\proj\\US Car Acciedent\\Python File\\usa_1.xlsx')

        data['Year']=data['Start_Time'].dt.year
        data['Month']=data['Start_Time'].dt.strftime('%b')
        data['Day']=data['Start_Time'].dt.day
        data['Hour']=data['Start_Time'].dt.hour
        data['Weekday']=data['Start_Time'].dt.strftime('%a')

        for col in ['TMC','Year','Day','Hour']:
            data[col] = data[col].astype('int16')

        for col in ["Severity", "Side", "City", "State", "County", "Timezone", "Airport_Code",
                "Wind_Direction", "Weather_Condition", "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight",
                "Astronomical_Twilight",'part_of_day','Month','Weekday','Year'] :

                data[col] = data[col].astype('category')

        data['Weekday'] = data['Start_Time'].dt.strftime('%a').apply( lambda x : 1 if x == 'Mon' else 2 if x == 'Tue'
                                              else 3 if x == 'Wed' else 4 if x == 'Thu' else 5 if x == 'Fri'
                                              else 6 if x == 'Sat' else 7).astype('int16')
        data['Month'] = data['Start_Time'].dt.month
        data['Year'] = data['Year'].astype('int16')


        numeric_columns = ['TMC', 'Start_Lat', 'Start_Lng', 'Distance(mi)', 'Temperature(F)',
                           'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                           'Precipitation(in)', 'Year', 'Month', 'Day', 'Hour', 'Weekday']

        categorical_columns = ['Airport_Code', 'Amenity', 'Bump', 'City', 'County', 'Crossing',
                               'Give_Way', 'Junction', 'Railway', 'Roundabout', 'Severity' , 'Side', 'State',
                               'Station', 'Stop', 'Street', 'Timezone', 'Traffic_Signal',
                               'Turning_Loop', 'Weather_Condition', 'Wind_Direction', 'part_of_day']

        # pipeline for categorical data

        imp = SimpleImputer(strategy="most_frequent")
        ms_c = CustomeScaler()

        categorical_pipeline = Pipeline( [ ('Categorical Features' , FeatureSelector(categorical_columns) ) , 
                                       ('Missing Value Treatement' , imp ) ,
                                       ('Categorical Encoding' , Encoding( categorical_columns ) ),
                                       ('Scaling Features' , ms_c )
                                     ]
                                   ) 

        # pipeline for Numerical data

        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        ms = CustomeScaler()

        numerical_pipeline = Pipeline( [ ('Numerical Features' , FeatureSelector(numeric_columns) ) , 
                                     ( 'Missing Value Treatement' , imputer ) ,
                                     ( 'Scaling Features' , ms)
                                     ]
                                   ) 

        # Combining two pipeline 

        preprocessing_pipeline = FeatureUnion( transformer_list = [ ( 'categorical_pipeline', categorical_pipeline ) ,
                                                                ( 'numerical_pipeline', numerical_pipeline )
                                                              ]
                                         )


        x = data.copy()
        y = data['Severity'].copy()

        test_size = 0.2
        x_train , x_test , y_train , y_test = train_test_split( x , y , test_size = test_size , random_state = 0 , stratify = y )

        preprocessing_pipeline.fit(x_train)
        
        filename = 'pipe.pickle'
        with open(filename, 'wb') as file:
            pickle.dump(preprocessing_pipeline, file)

        

if __name__ == '__main__':

    p = PreProcessing()
    
    p.train_pipe()

    
    
