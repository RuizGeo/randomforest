# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Random Forest Model
                                 A QGIS plugin
 Random Forest (RF)
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2018-03-19
        git sha              : $Format:%H$
        copyright            : (C) 2018 by Luis Fernando Chimelo Ruiz
        email                : ruiz.ch@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
import sys
#QGIS
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qgis.core import *
import numpy as np
import os
import sys
#import RF model
from .randomforestmodel import RandomForestModel

# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .randomforest_dialog import RandomForestDialog
from .to_evaluate import is_none, is_defined, exist_file,\
list_is_empty, txt_is_writable,field_is_integer,field_is_real,\
vector_is_readable,is_crs,is_join
import os.path


#Get layers QGIS
project = QgsProject.instance()

class RandomForest:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'AssessRFC_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = RandomForestDialog()

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&GeoPatterns')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'Random Forest')
        self.toolbar.setObjectName(u'Random Forest')

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('Random Forest', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/assess_rfc/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Random Forest'),
            callback=self.run,
            parent=self.iface.mainWindow())


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Random Forest'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar
        return 0

    def run(self):
        """Run method"""
        #Get plugin path
        self.plugin_path= QgsApplication.qgisSettingsDirPath()+'python/plugins/randomforest'
        #TabInput visble
        self.dlg.ui.tabWidget.setCurrentIndex(0)
        #Clear GUIs
        self.dlg.ui.comboBoxFieldTrain.clear()
        self.dlg.ui.comboBoxFieldVal.clear()
        self.dlg.ui.comboBoxCrit.clear()
        self.dlg.ui.comboBoxTrain.clear()
        self.dlg.ui.comboBoxVal.clear()
        self.dlg.ui.lineEditDataSet.clear()
        self.dlg.ui.lineEditOutModel.clear()
        self.dlg.ui.lineEditAssessFile.clear()
        #SetCheckState
        self.dlg.ui.checkBoxApplyModel.setCheckState(False)
        
        
        #enable
        self.dlg.ui.lineEditOutModel.setEnabled(False)
        self.dlg.ui.buttonOutModel.setEnabled(False)
        #Zero progressaBar
        self.dlg.ui.progressBar.setValue(0)
        #select radion button
        self.dlg.ui.radioButtonClass.setChecked(True)
        if self.dlg.ui.radioButtonClass.isChecked():
            #clear
            self.dlg.ui.comboBoxCrit.clear()
            #Set criterions Split comboBoxCrit
            self.dlg.ui.comboBoxCrit.addItems(['gini', 'entropy'])
        self.dlg.ui.radioButtonRegress.setChecked(True)
        if self.dlg.ui.radioButtonRegress.isChecked():
            #clear
            self.dlg.ui.comboBoxCrit.clear()
            #Set criterions Split comboBoxCrit
            self.dlg.ui.comboBoxCrit.addItems(['mse', 'mae'])
        #Creat Dict with layers names name:layer
        self.dict_layers={'None':None}
        #Insert fields comboBoxs
        self.fields={}
        #Dict with layers names
        dic_layers_qgis =  project.mapLayers()
        for name_layer in dic_layers_qgis.keys():
            #Append name and layer
            self.dict_layers[dic_layers_qgis[name_layer].name()]=dic_layers_qgis[name_layer]
            #print (name_layer)
            #assess if is vector
            if dic_layers_qgis[name_layer].type() == 0:
                 #Get field names
                 field_names = [field.name() for field in dic_layers_qgis[name_layer].dataProvider().fields() ]
                 self.fields[dic_layers_qgis[name_layer].name()]=field_names
                 #add name vectors
                 self.dlg.ui.comboBoxTrain.addItem(dic_layers_qgis[name_layer].name())    
                 self.dlg.ui.comboBoxVal.addItem(dic_layers_qgis[name_layer].name())
            else:
                pass

        
        
        #Insert values ComboBoxs                
        if self.dlg.ui.comboBoxTrain.count() == 0:   
            self.dlg.ui.comboBoxTrain.addItem('None')  
            self.dlg.ui.comboBoxFieldTrain.addItem('None') 
        else:
            #Get name layer
            names_fields=self.dlg.ui.comboBoxTrain.currentText()           
            #Get and field layer
            self.dlg.ui.comboBoxFieldTrain.addItems(self.fields[names_fields])
            
            
        if self.dlg.ui.comboBoxVal.count() == 0:             
            self.dlg.ui.comboBoxVal.addItem('None') 
            self.dlg.ui.comboBoxFieldVal.addItem('None')
        else:
            #Get name layer
            names_fields=self.dlg.ui.comboBoxVal.currentText() 
            self.dlg.ui.comboBoxFieldVal.addItems(self.fields[names_fields])

        print(self.plugin_path+os.sep+'packages_path.txt')
        #Append packages path in sys
        if self.dlg.ui.lineEditPackPath.text()=='':
            pass
        else:
            sys.path.append(self.dlg.ui.lineEditPackPath.text())
        #Read file packages_path.txt
        txt_packages_path=open(self.plugin_path+os.sep+'packages_path.txt','r')
        #Replace new line ('\n') for ''
        packages_path= [line.replace('\n', '') for line in txt_packages_path.readlines()]
        
        print(packages_path)
        #Set packages path in lineEdit
        if packages_path == []:
            pass
        else:
            self.dlg.ui.lineEditPackPath.setText(packages_path[0])
        #Close packages_path.txt
        txt_packages_path.close() 
        #Connect functions
        self.dlg.ui.buttonDataSet.clicked.connect(self.set_data_set)
        self.dlg.ui.buttonAssessFile.clicked.connect(self.set_assess_file)
        self.dlg.ui.buttonCancel.clicked.connect(self.cancel_GUI)
        self.dlg.ui.buttonRun.clicked.connect(self.run_model)
        self.dlg.ui.buttonOutModel.clicked.connect(self.set_model_path)
        #Connect functions RadionButton
        self.dlg.ui.radioButtonClass.clicked.connect(self.select_model_classification)        
        self.dlg.ui.radioButtonRegress.clicked.connect(self.select_model_regressor)         
        
        #Connect functions value changedspinBox
        self.dlg.ui.spinBoxStartEst.valueChanged.connect(self.value_changed_start_est)
        #self.dlg.ui.spinBoxStepSim.valueChanged.connect(self.value_changed_step_sim)
        self.dlg.ui.spinBoxStartDepth.valueChanged.connect(self.value_changed_start_depth)
        
        #Connect functions valueChanged comboBox
        self.dlg.ui.comboBoxTrain.currentIndexChanged.connect(self.value_changed_train)
        self.dlg.ui.comboBoxVal.currentIndexChanged.connect(self.value_changed_val)
        
        #Connect functions CheckBox
        self.dlg.ui.checkBoxApplyModel.stateChanged.connect(self.state_changed_apply_model)
        self.dlg.ui.checkBoxPackPath.stateChanged.connect(self.state_changed_packages_path)
        
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass
    def select_model_classification(self):
        print('Classification')
        #Clear comboBOx Criterion
        self.dlg.ui.comboBoxCrit.clear()
        #Add criterios
        self.dlg.ui.comboBoxCrit.addItems(['gini', 'entropy'])

    def select_model_regressor(self):
        print('Regressor')
        self.type_model = 'regressor'
        #Clear comboBOx Criterion
        self.dlg.ui.comboBoxCrit.clear()
        #Add criterios
        self.dlg.ui.comboBoxCrit.addItems(['mse', 'mae'])                
        
    def set_model_path(self):
        #clear lineEdit
        self.dlg.ui.lineEditOutModel.clear()
        #Save shape classification
        out_file = QFileDialog.getSaveFileName(None, self.tr('Save file'), None, " Shapefile (*.shp)")
        #get name and extension file
        name_file= out_file[0].split(os.sep)[-1]
        #Assess extension
        if out_file[-1].split('*')[-1].endswith('.shp)'):
            #insert extension
            if name_file.endswith('.shp'):
                self.dlg.ui.lineEditOutModel.setText(out_file[0])
            else:
                self.dlg.ui.lineEditOutModel.setText(out_file[0]+'.shp')
        else:
            pass
        
    def state_changed_apply_model(self):
        
        state=self.dlg.ui.checkBoxApplyModel.checkState()     
        print ('State apply mode: ',state)
        if state ==2:
            self.dlg.ui.lineEditOutModel.setEnabled(True)
            self.dlg.ui.buttonOutModel.setEnabled(True)
        else: 
            self.dlg.ui.lineEditOutModel.clear()
            self.dlg.ui.lineEditOutModel.setEnabled(False)
            self.dlg.ui.buttonOutModel.setEnabled(False)            

    def state_changed_packages_path(self):

        state=self.dlg.ui.checkBoxPackPath.checkState()     
        print ('State packages path: ',state)
        if state ==2:

            #Enable line edit
            self.dlg.ui.lineEditPackPath.setEnabled(True)

            
        else:            
            if self.dlg.ui.lineEditPackPath.text()=='':
                #Read file packages_path.txt
                txt_packages_path=open(self.plugin_path+os.sep+'packages_path.txt','w')
                txt_packages_path.write('')
                txt_packages_path.close()
                #self.dlg.ui.lineEditPackPath.setText('None')
                self.dlg.ui.lineEditPackPath.setEnabled(False)
                
            else:

                #Assess lineEdit and combobox data set
                if exist_file(self.dlg.ui.lineEditPackPath.text(),'Input: packages path is not exist'):
                    self.dlg.ui.lineEditPackPath.clear()
                    #Write file packages_path.txt
                    txt_packages_path=open(self.plugin_path+os.sep+'packages_path.txt','w')
                    txt_packages_path.write('')
                    txt_packages_path.close()
                    self.dlg.ui.checkBoxPackPath.setChecked(False) 
                    self.dlg.ui.lineEditPackPath.setEnabled(False)
                    return 0
                else:
                    #Write file packages_path.txt
                    txt_packages_path=open(self.plugin_path+os.sep+'packages_path.txt','w')
                    txt_packages_path.write(self.dlg.ui.lineEditPackPath.text())
                    txt_packages_path.close()
                    self.dlg.ui.lineEditPackPath.setEnabled(False)
                    #Set packages path in sys
                    sys.path.append(self.dlg.ui.lineEditPackPath.text())
                
                
            
                  
    def value_changed_train(self):
        
        #Clear
        self.dlg.ui.comboBoxFieldTrain.clear()      
        #Get name layer
        name_layer=self.dlg.ui.comboBoxTrain.currentText() 
        #Get field names
        #field_names = [field.name() for field in self.dict_layers[name_layer].dataProvider().fields() ]
        #Get and field layer
        self.dlg.ui.comboBoxFieldTrain.addItems(self.fields[name_layer])    
        print (self.fields[name_layer])
        return 0
    
    def value_changed_val(self):
        #Clear
        self.dlg.ui.comboBoxFieldVal.clear()      
        #Get name layer
        name_layer=self.dlg.ui.comboBoxVal.currentText() 
        #Get field names
        #field_names = [field.name() for field in self.dict_layers[name_layer].dataProvider().fields() ]
        #Get and field layer
        self.dlg.ui.comboBoxFieldVal.addItems(self.fields[name_layer])    
        print (self.fields[name_layer])
        return 0        
        
    def value_changed_start_est(self):
        #Set minimum valspinBoxEnd
        self.dlg.ui.spinBoxEndEst.setMinimum(self.dlg.ui.spinBoxStartEst.value())
        #Set maximum valspinBoxStep
        self.dlg.ui.spinBoxStepEst.setMaximum(self.dlg.ui.spinBoxEndEst.value())
        return 0
    
    def value_changed_start_depth(self):
        #Set minimum valspinBoxEndSim
        self.dlg.ui.spinBoxEndDepth.setMinimum(self.dlg.ui.spinBoxStartDepth.value())        
        #Set maximum valspinBoxStep
        self.dlg.ui.spinBoxStepDepth.setMaximum(self.dlg.ui.spinBoxEndDepth.value())        
        return 0
    def set_assess_file(self):
        #Clear
        self.dlg.ui.lineEditAssessFile.clear()
        #Save assess file
        assess_file=QFileDialog.getSaveFileName(None, self.tr('Save file'), None, " Text file (*.txt);;Comma-separated values (*.csv)")
        print(assess_file)
        #print (dir(assess_file[-1]))
        #get name and extension file
        name_file=assess_file[0].split(os.sep)[-1]
        #Assess extension
        if assess_file[0]=='':
            self.dlg.ui.lineEditAssessFile.clear()
            
        elif assess_file[-1].split('*')[-1].endswith('.txt)'):

            #insert extension
            if name_file.endswith('.txt'):
                self.dlg.ui.lineEditAssessFile.setText(assess_file[0])
            else:
                self.dlg.ui.lineEditAssessFile.setText(assess_file[0]+'.txt')
        else:
            if name_file.endswith('.csv'):
                self.dlg.ui.lineEditAssessFile.setText(assess_file[0])
            else:
                self.dlg.ui.lineEditAssessFile.setText(assess_file[0]+'.csv')
        
        
        return 0
    def set_data_set(self):
        #Open Directory
        self.pathSegs = QFileDialog.getExistingDirectory(None, self.tr('Open a folder'), None, QFileDialog.ShowDirsOnly)  
        #Set lineEditPathSegs
        self.dlg.ui.lineEditDataSet.setText(self.pathSegs)
        return 0
    
    def cancel_GUI(self):
        
        #disconnect       
        self.dlg.ui.buttonAssessFile.clicked.disconnect(self.set_assess_file)
        self.dlg.ui.buttonDataSet.clicked.disconnect(self.set_data_set)
        self.dlg.ui.buttonCancel.clicked.disconnect(self.cancel_GUI)
        self.dlg.ui.buttonRun.clicked.disconnect(self.run_model)
        self.dlg.ui.comboBoxTrain.currentIndexChanged.disconnect(self.value_changed_train)
        self.dlg.ui.comboBoxVal.currentIndexChanged.disconnect(self.value_changed_val)
        self.dlg.ui.checkBoxApplyModel.stateChanged.disconnect(self.state_changed_apply_model)
        self.dlg.ui.buttonOutModel.clicked.disconnect(self.set_model_path)

        self.dlg.close()
        
        return 0

        
      
    def run_model(self):
        #progressbar                
        self.dlg.ui.progressBar.setValue(0)
        print('Run model')
        print('Input and output files')

        #Assess lineEdit and combobox training samples
        if is_none(self.dlg.ui.comboBoxTrain.currentText(),'Input: Training samples'):
            return 0       
        #Assess lineEdit and combobox validation samples
        elif is_none(self.dlg.ui.comboBoxVal.currentText(),'Input: Validation samples'):
            return 0
        #Assess lineEdit and combobox data set
        elif is_defined(self.dlg.ui.lineEditDataSet.text(),'Input: data set is not defined'):
            return 0
        #Assess lineEdit and combobox data set
        elif exist_file(self.dlg.ui.lineEditDataSet.text(),'Input: data set is not exist'):
            return 0
        #Assess lineEdit and combobox Output text file
        elif is_defined(self.dlg.ui.lineEditAssessFile.text(),'Output: Text file is not defined'):
            return 0   

        else:
            print('To evaluate')
            #Model path
            model_path=self.dlg.ui.lineEditOutModel.text()
            #Get name and path layer - Input trainining samples
            name_train = self.dlg.ui.comboBoxTrain.currentText()
            path_train=self.dict_layers[name_train].dataProvider().dataSourceUri().split('|')[0]
            
            #Get name field class - training samples
            field_model_train = self.dlg.ui.comboBoxFieldTrain.currentText()
            #Type name field (train)
            fields_train=self.dict_layers[name_train].fields()
            type_field_train = fields_train[fields_train.indexFromName(field_model_train)].typeName()

            #Get name and path layer - Input validation samples
            name_val=self.dlg.ui.comboBoxVal.currentText()
            path_val=self.dict_layers[name_val].dataProvider().dataSourceUri().split('|')[0]
            #Get name field class - training samples
            field_model_val = self.dlg.ui.comboBoxFieldVal.currentText()
            #Type name field (Val)
            fields_val=self.dict_layers[name_val].fields()
            type_field_val = fields_val[fields_val.indexFromName(field_model_val)].typeName()
            #To evaluate radioButtonClas
            if self.dlg.ui.radioButtonClass.isChecked():
                #Set value type_model
                type_model='classification'
                #Field must be integer
                if field_is_integer(type_field_train,'Input: Training classes field is not integer')==False:
                    return 0
                #Field must be integer
                if field_is_integer(type_field_val,'Input: Validation classes field is not integer')==False:
                    return 0
                print('RadionButton classification checked')
            else:
                #set type model
                type_model='regression'

                #Field must be integer
                if field_is_real(type_field_train,'Input: Training classes field is not real')==False:
                    return 0
                #Field must be integer
                if field_is_real(type_field_val,'Input: Validation classes field is not real')==False:
                    return 0            
            #data set path
            dataset_path=self.dlg.ui.lineEditDataSet.text() 
            #Values n_estimators
            start_est,end_est,step_est = self.dlg.ui.spinBoxStartEst.value(),\
            self.dlg.ui.spinBoxEndEst.value(),self.dlg.ui.spinBoxStepEst.value()
            #values max_depth
            start_dp,end_dp,step_dp=self.dlg.ui.spinBoxStartDepth.value(),\
            self.dlg.ui.spinBoxEndDepth.value(),self.dlg.ui.spinBoxStepDepth.value()
            #Get criterion
            criterion_split = self.dlg.ui.comboBoxCrit.currentText()
            #Assess exist files training samples
            if exist_file(path_train,'Input: Training samples is not exist'):
                return 0
            if vector_is_readable(path_train,'Input: Training samples is not readable') == False:
                return 0
            #Assess exist files training samples
            if exist_file(path_val,'Input: Validation samples is not exist'):
                return 0 
            
            if vector_is_readable(path_val,'Input: Validation samples is not readable') == False:
                return 0
            #Get names files segmentations
            segs_names=[f for f in os.listdir(dataset_path)  if f.endswith('.shp')]
            if  list_is_empty(segs_names, 'Input: Data set folder is empty'):
                return 0  
            
            #Get path assess file
            path_assess_file=self.dlg.ui.lineEditAssessFile.text()
            if is_defined(path_assess_file,'Output: Text file is not defined'):
                return 0
            #Get path assess file
            if txt_is_writable(path_assess_file,'Output: Text file is not exist'):
                return 0
                  
           #To avaliate checkBox Apply Model
            if self.dlg.ui.checkBoxApplyModel.checkState()==2:
                state_checkBox_model= 'True'
                #Assess   defined vector file
                if is_defined(model_path,'Output: vector file is not defined'):
                    return 0
            else:
                state_checkBox_model= 'False'
                model_path='/'
            

            #progressbar
            self.dlg.ui.progressBar.setValue(10)

            #Run Assess RFC
            RandomForestModel(self.dlg.ui.progressBar,path_train,dataset_path,\
                    path_val,start_est,end_est,step_est,\
                    start_dp,end_dp,step_dp,field_model_train,\
                    field_model_val,criterion_split,path_assess_file,state_checkBox_model,model_path,type_model)
            print('Finish')
            #progressbar                
            self.dlg.ui.progressBar.setValue(100)
            #self.dlg.ui.progressBar.setTextVisible(True)
            #self.dlg.ui.progressBar.setFormat('Finish')
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Finish')
            msg.setWindowTitle("Info")
            msg.exec_() 
   

            
            
