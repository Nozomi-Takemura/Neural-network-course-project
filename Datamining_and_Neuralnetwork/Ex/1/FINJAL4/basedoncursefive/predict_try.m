function Yfit = predict_try(this,X,varargin)
        %PREDICT Predict response of the model.
        %   YFIT=PREDICT(MODEL,X) returns predicted response YFIT for linear model
        %   MODEL and predictors X. Pass X as a matrix with P columns, where P is the
        %   number of predictors used for training. YFIT is an N-by-L matrix of the
        %   same type as Y, where N is the number of observations (rows) in X
        %   and L is the number of values in Lambda.
        %
        %   YFIT=(MODEL,X,'PARAM1',val1,'PARAM2',val2,...) specifies optional
        %   parameter name/value pairs:
        %       'ObservationsIn'    - String specifying the data orientation,
        %                             either 'rows' or 'columns'. Default: 'rows'
        %                           NOTE: Passing observations in columns can
        %                                 significantly speed up prediction.
        %
        %   See also RegressionLinear, Lambda.

            % Handle input data such as "tall" requiring a special adapter
            adapter = classreg.learning.internal.makeClassificationModelAdapter(this,X,varargin{:});
            if ~isempty(adapter)            
                Yfit = predict(adapter,X,varargin{:});
                return
            end
        
            internal.stats.checkSupportedNumeric('X',X,false,true);
            
            % Detect the orientation
            obsIn = internal.stats.parseArgs({'observationsin'},{'rows'},varargin{:});
            obsIn = validatestring(obsIn,{'rows' 'columns'},...
                'classreg.learning.internal.orientX','ObservationsIn');
            obsInRows = strcmp(obsIn,'rows');
            
            % Predictions for empty X
            if isempty(X)
                D = numel(this.PredictorNames);
                if obsInRows
                    str = getString(message('stats:classreg:learning:regr:RegressionModel:predictEmptyX:columns'));
                    Dpassed = size(X,2);
                else
                    Dpassed = size(X,1);
                    str = getString(message('stats:classreg:learning:regr:RegressionModel:predictEmptyX:rows'));
                end
                if Dpassed~=D
                    error(message('stats:classreg:learning:regr:RegressionModel:predictEmptyX:XSizeMismatch', D, str));
                end
                Yfit = NaN(0,1);
                return;
            end

            % Predict
            Yfit = this.PrivResponseTransform(response(this,X,obsInRows));
        end