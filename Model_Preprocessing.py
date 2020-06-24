from sklearn.base import BaseEstimator, TransformerMixin 
import pandas as pd
import joblib
#Custom Transformer that extracts columns passed as argument to its constructor x`

class FeatureSelector( BaseEstimator, TransformerMixin ):

    def __init__( self, feature_names ):
        self._feature_names = feature_names 
     
    def fit( self, X, y = None ):
        return self 

    def transform( self, X, y = None ):
        return X[ self._feature_names ]
    
class Encoding(BaseEstimator, TransformerMixin):
    def __init__(self , categorical_columns ):
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
    

def pipe():
    pipe = joblib.load('pipe.joblib')
    return pipe

if __name__ == '__main__':
    FeatureSelector.__module__ = "__main__"
    Encoding.__module__ = "__main__"
    pipe()