# From https://github.com/IridiumWasTaken/pumpy3

import serial
import logging
import re
import threading
from time import sleep


# def remove_crud(string):
#     """Return string without useless information.
#      Return string with trailing zeros after a decimal place, trailing
#      decimal points, and leading and trailing spaces removed.
#      """
#     if "." in string:
#         string = string.rstrip('0')
#
#     string = string.lstrip('0 ')
#     string = string.rstrip(' .')
#
#     return string


def convert_units(val, fromUnit, toUnit):
    # TODO: Used anywhere?
    """ Convert flowrate units. Possible volume values: ml, ul, pl; possible time values: hor, min, sec
    :param fromUnit: unit to convert from
    :param toUnit: unit to convert to
    :type fromUnit: str
    :type toUnit: str
    :return: float
    """
    time_factor_from = 1
    time_factor_to = 1
    vol_factor_to = 1
    vol_factor_from = 1

    if fromUnit[-3:] == "sec":
        time_factor_from = 60
    elif fromUnit == "hor":  # does it really return hor?
        time_factor_from = 1 / 60
    else:
        pass

    if toUnit[-3:] == "sec":
        time_factor_to = 1 / 60
    elif toUnit[-3:] == "hor":
        time_factor_to = 60
    else:
        pass

    if fromUnit[:2] == "ml":
        vol_factor_from = 1000
    elif fromUnit[:2] == "nl":
        vol_factor_from = 1 / 1000
    elif fromUnit[:2] == "pl":
        vol_factor_from = 1 / 1e6
    else:
        pass

    if toUnit[:2] == "ml":
        vol_factor_to = 1 / 1000
    elif toUnit[:2] == "nl":
        vol_factor_to = 1000
    elif toUnit[:2] == "pl":
        vol_factor_to = 1e6
    else:
        pass

    return val * time_factor_from * time_factor_to * vol_factor_from * vol_factor_to


def convert_str_units(abbr):
    # TODO: Used anywhere?
    """ Convert string units from serial units m, u, p and s, m, h to full strings.
    :param abbr: abbreviated unit
    :type abbr: str
    :return: str
    """
    first_part = abbr[0] + "l"
    if abbr[2] == "s":
        second_part = "sec"
    elif abbr[2] == "m":
        second_part = "min"
    elif abbr[2] == "h":
        second_part = "hor"  # is that true?
    else:
        raise ValueError("Unknown unit")

    resp = first_part + "/" + second_part
    return resp


class Chain(serial.Serial):
    """Create Chain object.
    Harvard syringe pumps are daisy chained together in a 'pump chain'
    off a single serial port. A pump address is set on each pump. You
    must first create a chain to which you then add Pump objects.
    Chain is a subclass of serial.Serial. Chain creates a serial.Serial
    instance with the required parameters, flushes input and output
    buffers (found during testing that this fixes a lot of problems) and
    logs creation of the Chain. Adapted from pumpy on github.
    """

    def __init__(self, port, baudrate=115200):
        """
        :param port: Port of pump at PC
        :type port: str
        """
        serial.Serial.__init__(self, port=port, stopbits=serial.STOPBITS_TWO,
                               parity=serial.PARITY_NONE,
                               bytesize=serial.EIGHTBITS, xonxoff=False,
                               baudrate=baudrate, timeout=2)
        self.flushOutput()
        self.flushInput()
        logging.info('Chain created on %s', port)


class Pump:
    """Create Pump object for Harvard Pump.
        Argument:
            Chain: pump chain
        Optional arguments:
            address: pump address. Default is 0.
            name: used in logging. Default is PhD2000. NOTE MODEL 22 OR 44 PROTOCOL!!
        """

    def __init__(self, chain, address=00, name='PhD2000'):
        self.name = name
        self.serialcon = chain
        self.address = '{0:02.0f}'.format(address)
        self.diameter = None
        self.flowrate = None
        self.targetvolume = None
        self.state = None

        """Query model and version number of firmware to check pump is
        OK. Responds with a load of stuff, but the last three characters
        are XXY, where XX is the address and Y is pump status. :, > or <
        when stopped, running forwards, or running backwards. Confirm
        that the address is correct. This acts as a check to see that
        the pump is connected and working."""
        try:
            self.write('VER')
            resp = self.read(17)

            if 'PHD' not in resp:
                raise PumpError('No response from pump at address %s' %
                                self.address)

            if resp[-1] == ':':
                self.state = 'idle'
                print(self.state)
            elif resp[-1] == '>':
                self.state = 'infusing'
                print(self.state)
            elif resp[-1] == '<':
                self.state = 'withdrawing'
                print(self.state)
            elif resp[-1] == '*':
                self.state = 'stalled'
                print(self.state)
            else:
                raise PumpError('%s: Unknown state encountered' % self.name)

        except PumpError:
            self.serialcon.close()
            raise

        logging.info('%s: created at address %s on %s', self.name,
                     self.address, self.serialcon.port)

    def __repr__(self):
        string = ''
        for attr in self.__dict__:
            string += '%s: %s\n' % (attr, self.__dict__[attr])
        return string

    def write(self, command):
        """ Write serial command to pump.
        :param command: command to write
        :type command: str
        """
        self.serialcon.write((self.address + command + '\r').encode())

    def read(self, bytes=5):
        """ Read serial stream from pump.
        :param bytes: number of bytes to read
        :type bytes: int
        :return: str
        """
        response = self.serialcon.read(bytes)

        if len(response) == 0:
            pass
            # raise PumpError('%s: no response to command' % self.name)
        else:
            response = response.decode()
            response = response.replace('\n', '')
            return response

    def waituntilfinished(self):
        """ Try to read pump state and return it. """
        while self.state == "infusing" or self.state == "withdrawing":
            try:
                self.write('VOL')
                resp = self.read(15)
                # resp = self.read(5)
                if ':' in resp:
                    self.state = "idle"
                    return "finished"
            except:
                pass

    def run(self):
        self.write('RUN')
        resp = self.read(17)

        self._errorcheck(resp)

        self.state = 'infusing'

    def rev(self):
        self.write('REV')
        resp = self.read(17)

        self._errorcheck(resp)

        self.state = 'withdrawing'

    def infuse(self):
        self.run()

        if self.state == 'withdrawing':
            self.stop()
            self.rev()

    def withdraw(self):
        self.rev()

        if self.state == 'infusing':
            self.stop()
            self.run()

    def stop(self):
        self.write('STP')
        resp = self.read(17)

        self._errorcheck(resp)

        sleep(0.1)
        if self.state == 'infusing' or self.state == 'withdrawing':
            raise PumpError('%s: Pump could not be stopped.' % self.name)

    def _errorcheck(self, resp):
        if resp[-1] == ':':
            self.state = 'idle'
            print(self.state)
        elif resp[-1] == '>':
            self.state = 'infusing'
            print(self.state)
        elif resp[-1] == '<':
            self.state = 'withdrawing'
            print(self.state)
        elif resp[-1] == '*':
            self.state = 'stalled'
            print(self.state)
        else:
            raise PumpError('%s: Unknown state encountered' % self.name)

    def clear_accumulated_volume(self):
        self.write('CLV')
        resp = self.read(17)

        self._errorcheck(resp)

    def clear_target_volume(self):
        self.write('CLT')
        resp = self.read(17)

        self._errorcheck(resp)

    def set_rate(self, flowrate, units):
        flowrate_str = "%4.4f" % flowrate
        if units == 'm/m':
            write_str = 'MLM'
        elif units == 'u/m':
            write_str = 'ULM'
        elif units == 'm/h':
            write_str = 'MLH'
            self.rate_units = "ml/h"
        elif units == 'u/h':
            write_str = 'ULH'
        else:
            raise PumpError('%s: Unknown unit specified' % self.name)

        self.write(write_str + flowrate_str)
        resp = self.read(17)
        self._errorcheck(resp)

    def setdiameter(self, diameter):
        # TODO: Limit range of values
        self.write('MMD' + str(diameter))
        resp = self.read(17)
        self._errorcheck(resp)

    def settargetvolume(self, volume):
        """ Set target volume in mL. """
        self.write('MLT' + str(volume))
        resp = self.read(17)
        self._errorcheck(resp)

    def getdiameter(self):
        self.write('DIA')
        resp = self.read(17)

        self._errorcheck(resp)
        matches = re.search(r"(\d+\.?\d*)", resp)
        if matches is not None:
            return matches.group(1) + " mm"
        else:
            raise PumpError('%s: Unknown answer received' % self.name)

    def getrate(self):
        self.write('RAT')
        resp = self.read(19)

        self._errorcheck(resp)
        matches = re.search(r"(\d+\.?\d*)", resp)
        if matches is not None:
            self.write('RNG')
            resp = self.read(17)
            self._errorcheck(resp)
            return matches.group(1) + " " + resp[:4]
        else:
            raise PumpError('%s: Unknown answer received' % self.name)

    def ivolume(self):
        self.write('VOL')
        resp = self.read(17)

        self._errorcheck(resp)
        matches = re.search(r"(\d+\.?\d*)", resp)
        if matches is not None:
            return matches.group(1) + " " + "ml"
        else:
            raise PumpError('%s: Unknown answer received' % self.name)

    def gettargetvolume(self):
        self.write('TAR')
        resp = self.read(17)

        self._errorcheck(resp)
        matches = re.search(r"(\d+\.?\d*)", resp)
        if matches is not None:
            return matches.group(1) + " " + "ml"
        else:
            raise PumpError('%s: Unknown answer received' % self.name)


class PumpError(Exception):
    pass
