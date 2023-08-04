import os
import numpy as np
import casacore.tables as tables


class ReadMS:
    '''
    This class takes an input .ms file and reads it via casacore.tables

    See each method for further information of its functionality

    Methods:
        __init__()                               : returns "None"
        GetMainTableData(columnName, range=None) : returns a numpy array of the data column required. Parameter "range" is a tuple specifying the range of data to be returned. The default value of "range" is "None", returning all data of the respective column
        ListSubTables()                          : returns a list of the names of the tables containing metadata (not visible within the main table)
        GetFrequencyChannels()                   : returns "None". Updates the class attribute self.frequencyChannels by assigning to it a numpy array containing the frequency channels
        GetPolarisationData()                    : returns a numpy array of the polarization data describing the measurement set
        ConvertUVWToWavelengths()                : returns "None". Updates the class attributes self.u, self.v, and self.w with the respective coordinates in wavenumbers, at each frequency channel independently
        PlotSamplingFunction()                   : returns "None". Generates and saves a figure of the 2D sampling function S = f(u,v) on a UV plane

    Attributes:
        mainTable         : stores the main table from the .ms file (using casacore functionality)
        frequencyTable    : stores the subtable incorporating the frequency channels (using casacore functionality)
        polarizationTable : stores the subtable incorporating the polarization data (using casacore functionality)
        antennaTable      : stores the subtable incorporating the antennas data (using casacore functionality)
        u                 : intended to store u coordinates in wave numbers, for each frequency channel independently
        v                 : intended to store v coordinates in wave numbers, for each frequency channel independently
        w                 : intended to store w coordinates in wave numbers, for each frequency channel independently
        frequencyChannels : intended to store the freqnency channels of the measurement set
    '''

    def __init__(self, inputFile):

        '''
        This is the initialiser method of the class

        inputFile : takes a string pointing to the address of the .ms file in the directory

        Class is initialised as follows:

        - input .ms file is provided and read using the casacore.tables method
        - metadata including frequency channels and polirizations are also read through casacore
        - for frequency and polarization data, it is assumed that the relevant metadata tables are named as per convention (i.e., "SPECTRAL_WINDOW" and "POLARIZATION", respectively)
        - essential inter-method parameters are initialised here as empty or None variables, as follows:

            u, v, w initialised as empty dictionaries, with the intention to append into them relevant coordinates at each frequency channel independently.
                (e.g., u['0'] = {FREQ CHAN=0})

            frequency channels are initialised as None variable, as they will be used in more than one function.

        The __init__ method returns None
        '''

        self.mainTable = tables.table(inputFile, readonly=False)
        self.frequencyTable = tables.table(os.path.join(inputFile, 'SPECTRAL_WINDOW'))
        self.polarizationTable = tables.table(os.path.join(inputFile, 'POLARIZATION'))
        self.antennaTable = tables.table(os.path.join(inputFile, 'ANTENNA'))
        self.fieldTable = tables.table(os.path.join(inputFile, 'FIELD'))

        self.u = {}
        self.v = {}
        self.w = {}
        self.frequencyChannels = None

        return None

    def GetMainTableData(self, columnName, range=None):

        '''
        columnName : takes a string defining the name of the column whose data are required
        range=None : (optional) takes a tuple of length 2 defining the slice of data to be returned (initial and final index values). Default value returns the full dataset

        This method extracts data tied to a specific column in the main table of the MeasurementSet
        It uses casacore functionality for this purpose
        '''

        output = tables.tablecolumn(self.mainTable, columnName)

        if range is None:
            return np.array(output[:])
        else:
            return np.array(output[range[0]: range[1]])

    def GetFreqTableData(self, columnName):

        """
                columnName : takes a string defining the name of the column whose data are required

                This method extracts data tied to a specific column in the main table of the MeasurementSet
                It uses casacore functionality for this purpose

        """

        output = tables.tablecolumn(self.frequencyTable, columnName)
        return output

    def GetAntennaTableData(self, columnName):

        '''
        columnName : takes a string defining the name of the column whose data are required

        This method extracts data tied to a specific column in the main table of the MeasurementSet
        It uses casacore functionality for this purpose

        '''

        output = tables.tablecolumn(self.antennaTable, columnName)
        return output

    def GetFieldTableData(self, columnName):


        '''
        columnName : takes a string defining the name of the column whose data are required

        This method extracts data tied to a specific column in the main table of the MeasurementSet
        It uses casacore functionality for this purpose

        '''

        output = tables.tablecolumn(self.fieldTable, columnName)
        return output


    def ListSubTables(self):

        '''
        No arguments taken

        This method returns a list of strings, detailing the names of the various subtables of the measurement set
        '''

        output = []

        for item in self.mainTable.getsubtables():
            output.append(str(os.path.basename(item)))

        return output

    def UpdateFrequencyChannels(self):

        '''
        No arguments taken

        This method returns "None"

        It updates the class attribute "self.frequencyChannels", by equating it to a numpy array bearing the various frequency channels of the measurement set
        It uses casacore functionality to access the relevant subtable data
        It assumes that the column pointing to frequency channels is named as per convention (i.e., "CHAN_FREQ")
        '''

        self.frequencyChannels = tables.tablecolumn(self.frequencyTable, 'CHAN_FREQ')
        self.frequencyChannels = np.array(self.frequencyChannels[0])

        return None

    def GetPolarizationData(self):

        '''
        No arguments taken

        This method returns a numpy array bearing the polarization data of the measurement set

        It uses casacore functionality to access the relevant subtable data
        It assumes that the column pointing to polarization data is named as per convention (i.e., "CORR_TYPE")
        '''

        output = tables.tablecolumn(self.polarizationTable, 'CORR_TYPE')

        return np.array(output[0])

    def ConvertUVWToWavelengths(self):

        '''
        No arguments taken

        This method returns "None"

        It updates the class attribites "self.u", "self.v", and "self.w"
        '''

        # updating "self.frequencyChannels" array if necessary
        if self.frequencyChannels is None:
            self.GetFrequencyChannels()

        # extracting raw uvw coordinates from the measurement set
        uvwCoordinates = np.array(self.GetMainTableData('UVW'))

        # appending into u, v, and w the baseline distances in wave numbers. Values corresponding to each frequency channel are appended into separate arrays
        for n in range(len(self.frequencyChannels)):
            self.u[str(n)] = (uvwCoordinates[:, 0] * self.frequencyChannels[n]) / 299792458.
            self.v[str(n)] = (uvwCoordinates[:, 1] * self.frequencyChannels[n]) / 299792458.
            self.w[str(n)] = (uvwCoordinates[:, 2] * self.frequencyChannels[n]) / 299792458.

        return None


    def write_maintable_column(self, column_name, value):

        self.mainTable.putcol(column_name, value)


    # def PlotSamplingFunction(self, freqChannel='all', include_w=False):
    #
    #     '''
    #     freqChannel=all : (optional) takes an integer, pointing at the range of frequency channel indices from 0, which should be plotted collectively on the UV plane. Default is "all", which plots values for all frequency channels included in the measurement set
    #     include_w=False : (optional) takes a boolean, but is not currently functional in the method (i.e., it is indifferent/it achieves nothing). This parameter is intended to tell the method whether to construct a 3D plot incorporating the sampling function scattered in a uvw space
    #
    #     This method returns "None"
    #
    #     It plots S(u,v) = 1 at the locations corresponding to the values stored in u and v at the requested frequency channels
    #     It also plots S(-u, -v) = 1, which are not usually stored due to symmetry
    #     It saves the generated plot into a directory ./Results as a .png file
    #     '''
    #     # updating "self.u", "self.v", and "self.w" dictionaries if necessary
    #     if self.u == {}:
    #         self.ConvertUVWToWavelengths()
    #
    #     # handling freqChannel='all'
    #     if freqChannel == 'all':
    #         freqChannel = len(self.frequencyChannels)
    #
    #     # raising error if requested number of channels is higher than the available number of channels
    #     elif freqChannel > len(self.frequencyChannels):
    #         raise IndexError('The requested range of frequency channel values is out of bounds.')
    #
    #     # generating the plot (via matplotlib)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #
    #     for n in range(freqChannel):
    #         ax.scatter(
    #             self.u[str(n)],
    #             self.v[str(n)],
    #             np.ones(len(self.v[str(n)])),
    #             'k',
    #         )
    #         ax.scatter(
    #             -self.u[str(n)],
    #             -self.v[str(n)],
    #             np.ones(len(self.v[str(n)])),
    #             'k',
    #         )
    #
    #     plt.xlabel('u - wavenumbers')
    #     plt.ylabel('v - wavenumbers')
    #     plt.title('2D sampling function on a UV plane')
    #
    #     # assigning the directory to save into, ans saving the figure
    #     path = os.path.join(os.getcwd(), 'Results')
    #     if not os.path.exists(path):
    #         os.mkdir(path)
    #     plt.savefig(os.path.join(path, 'channels_' + str(freqChannel) + '.png'))
    #     plt.close()
    #
    #     return None


if __name__ == '__main__':
    print('Please run main.py')

